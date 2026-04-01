#!/bin/bash
# test_traffic_dataset.sh — 5G network-traffic QoS/QoE 진단 LLM 평가 + 검증
#
# 사용법:
#   ./test_traffic_dataset.sh [llama|deepseek|qwen] [옵션]
#
# 옵션:
#   --max N              : 평가할 최대 레코드 수 (기본: 10)
#   --split train|val|test: 데이터 분할 태그 (기본: test, 출력 파일명에만 반영)
#   --run-pipeline       : Python 진단 파이프라인을 먼저 실행해 JSONL 재생성
#   --api generate|chat  : 사용할 API 엔드포인트 (기본: chat)
#   --log-file FILE      : 장시간 실행 로그 저장 경로
#
# 흐름:
#   1) 서버 상태 확인 (healthz)
#   2) [선택] network_traffic_dataset.py --max N --no-example-tests 실행
#      → network_traffic_diagnosis.jsonl 생성 (llm_prompt 필드 포함)
#   3) diagnosis JSONL 읽기 → llm_prompt 추출 또는 동적 구성
#   4) LLM 서버 호출 (/v1/chat/completions 또는 /generate)
#   5) 응답 파싱 + rule-engine 결과(severity_level, inferred_root_cause) 대비 검증
#   6) 벤치마크 리포트 출력
#
# 의존성:
#   - jq          : JSON 처리
#   - curl        : HTTP 호출
#   - 실행 중인 SLM 추론 서버 (./test_slm.sh llama|deepseek|qwen 으로 기동)
#
# safe_ops 빌드 (선택 — Python 스크립트 내에서 자동 폴백):
#   cd jetson_slm_stack/app/csrc && pip install -e .
#
# 예시:
#   ./test_traffic_dataset.sh llama
#   ./test_traffic_dataset.sh llama --max 20 --run-pipeline
#   ./test_traffic_dataset.sh deepseek --max 50 --api generate
#   ./test_traffic_dataset.sh llama --split test --api chat --max 30
#   ./test_traffic_dataset.sh llama --run-pipeline --log-file test_slm_output/traffic.log

set -uo pipefail

# ── 인수 파싱 ──────────────────────────────────────────────────────────────────
MODEL_TYPE="${1:-}"
if [[ "$MODEL_TYPE" != "llama" && "$MODEL_TYPE" != "deepseek" && "$MODEL_TYPE" != "qwen" ]]; then
    echo "Usage: ./test_traffic_dataset.sh [llama|deepseek|qwen]"
    echo "       [--max N] [--run-pipeline] [--api generate|chat] [--log-file FILE]"
    exit 1
fi
shift

MAX_SAMPLES=10
API_MODE="chat"
SPLIT="test"
RUN_PIPELINE=false
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max)           MAX_SAMPLES="$2";  shift 2 ;;
        --split)         SPLIT="$2";        shift 2 ;;
        --api)           API_MODE="$2";     shift 2 ;;
        --run-pipeline)  RUN_PIPELINE=true; shift 1 ;;
        --log-file)      LOG_FILE="$2";     shift 2 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$SPLIT" != "train" && "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "[ERROR] --split must be 'train', 'val', or 'test'"
    exit 1
fi

if [[ "$API_MODE" != "generate" && "$API_MODE" != "chat" ]]; then
    echo "[ERROR] --api must be 'generate' or 'chat'"
    exit 1
fi

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/jetson_slm_stack"
DIAG_SCRIPT="$COMPOSE_DIR/dataset/scripts/network_traffic_dataset.py"
DIAG_JSONL="$COMPOSE_DIR/dataset/prepared/network_traffic/network_traffic_diagnosis.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/test_slm_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUTPUT_DIR/traffic_eval_${MODEL_TYPE}_${SPLIT}_${TIMESTAMP}.jsonl"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"

case "$MODEL_TYPE" in
    llama)    PORT=8000 ;;
    deepseek) PORT=8001 ;;
    qwen)     PORT=8002 ;;
esac

mkdir -p "$OUTPUT_DIR"

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="${OUT_FILE%.jsonl}.log"
fi
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

API_PATH="$([[ "$API_MODE" == "chat" ]] && echo "v1/chat/completions" || echo "generate")"
EFFECTIVE_MAX_NEW_TOKENS=256

# ── 배너 ───────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Traffic Diagnosis Eval: $MODEL_TYPE  (port $PORT)"
  echo "  Split: $SPLIT   Max: $MAX_SAMPLES   API: /$API_PATH"
echo "════════════════════════════════════════════════"

# ── 서버 상태 확인 ─────────────────────────────────────────────────────────────
if ! curl -sf "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
    echo ""
    echo "[ERROR] 서버가 실행 중이지 않습니다 (port $PORT)"
    echo "  먼저 실행: ./test_slm.sh $MODEL_TYPE"
    exit 1
fi

HEALTH=$(curl -s "http://localhost:$PORT/healthz")
MODEL_ID_VAL=$(echo "$HEALTH" | jq -r '.model_id              // "N/A"')
MEM_TOTAL=$(   echo "$HEALTH" | jq -r '.cuda_memory_total      // 0')
MEM_ALLOC=$(   echo "$HEALTH" | jq -r '.cuda_memory_allocated  // 0')
MEM_TOTAL_MB=$(echo "$MEM_TOTAL" | awk '{printf "%.1f", $1/1024/1024}')
MEM_ALLOC_MB=$(echo "$MEM_ALLOC" | awk '{printf "%.1f", $1/1024/1024}')

echo ""
echo "  Model : $MODEL_ID_VAL"
echo "  VRAM  : ${MEM_ALLOC_MB} MB / ${MEM_TOTAL_MB} MB"

# ── [선택] Python 진단 파이프라인 실행 ────────────────────────────────────────
if [[ "$RUN_PIPELINE" == true ]]; then
    echo ""
    echo "[pipeline] network_traffic_dataset.py --max $MAX_SAMPLES ..."
    if [[ -f "$VENV_ACTIVATE" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
    fi
    python "$DIAG_SCRIPT" --max "$MAX_SAMPLES" --no-example-tests
    echo "[pipeline] done."
fi

if [[ ! -f "$DIAG_JSONL" ]]; then
    echo ""
    echo "[ERROR] Diagnosis JSONL not found: $DIAG_JSONL"
    echo "  Solution A: ./test_traffic_dataset.sh $MODEL_TYPE --run-pipeline"
    echo "  Solution B: python $DIAG_SCRIPT --max $MAX_SAMPLES --no-example-tests"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$DIAG_JSONL")
if [[ "$MAX_SAMPLES" -gt 0 && "$MAX_SAMPLES" -lt "$TOTAL_LINES" ]]; then
    N_SAMPLES=$MAX_SAMPLES
else
    N_SAMPLES=$TOTAL_LINES
fi

echo "  Diag  : $DIAG_JSONL"
echo "  Samples: $N_SAMPLES / $TOTAL_LINES"
echo "  Output: $OUT_FILE"
echo "  Log   : $LOG_FILE"
echo ""

# ── JSON utilities ─────────────────────────────────────────────────────────────
is_json()  { jq -e . >/dev/null 2>&1 <<< "$1"; }
json_read() {
    local body="$1" filter="$2" fallback="$3"
    if is_json "$body"; then
        jq -r --arg fb "$fallback" "$filter // \$fb" <<< "$body"
    else
        echo "$fallback"
    fi
}

# ── LLM system prompt (mirrors LLM_SYSTEM_PROMPT in the Python script) ─────────
LLM_SYS="You are a 5G network QoS/QoE diagnostics assistant. \
Given network KPI measurements and a pre-computed rule analysis, \
respond EXACTLY in this format (no extra text):\n\
severity: <normal|warning|critical>\n\
root_cause: <radio_coverage_degradation|radio_retainability_failure|\
cell_edge_or_mobility_instability|core_or_backhaul_congestion|\
transport_or_core_issue|device_side_factor|multi_domain_degradation|indeterminate>\n\
action: <one concise action recommendation>"

# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────
IDX=0
SUCCESS=0
FAIL=0
SEV_MATCH=0
RC_MATCH=0
LATENCIES=""
THROUGHPUTS=""
PROMPT_TOKS=0
COMP_TOKS=0

while IFS= read -r line && [[ $IDX -lt $N_SAMPLES ]]; do
    # 빈 줄 건너뛰기
    [[ -z "${line//[[:space:]]/}" ]] && continue

    # 파싱 불가 JSON 건너뛰기
    if ! is_json "$line"; then
        echo "[WARN] skip malformed JSON"
        continue
    fi

    TS=$(          echo "$line" | jq -r '.timestamp           // "unknown"')
    SEVERITY_RULE=$(echo "$line" | jq -r '.severity_level      // ""')
    RC_RULE=$(      echo "$line" | jq -r '.inferred_root_cause // ""')
    LLM_PROMPT=$(   echo "$line" | jq -r '.llm_prompt          // ""')

    # llm_prompt 필드가 없는 경우 (이전 pipeline 버전 산출물) -- 기본 프롬프트 구성
    if [[ -z "$LLM_PROMPT" || "$LLM_PROMPT" == "null" ]]; then
        BREACHED=$(echo "$line" | jq -c '.breached_kpis // []')
        LLM_PROMPT="5G network record (ts=$TS). Rule severity=$SEVERITY_RULE. Breached KPIs: $BREACHED. Diagnose."
    fi

    IDX=$((IDX + 1))
    printf "[%s][%3d/%d] ts=%-22s sev=%-8s rc=%-36s " \
        "$(date +%H:%M:%S)" "$IDX" "$N_SAMPLES" \
        "${TS:0:22}" "$SEVERITY_RULE" "${RC_RULE:0:36}"

    # ── API 호출 ──────────────────────────────────────────────────────────────
    if [[ "$API_MODE" == "chat" ]]; then
        MSG_JSON=$(jq -n \
            --arg s "$LLM_SYS" \
            --arg u "$LLM_PROMPT" \
            '[{role:"system",content:$s},{role:"user",content:$u}]')
        PAYLOAD=$(jq -n \
            --argjson messages       "$MSG_JSON" \
            --argjson max_new_tokens "$EFFECTIVE_MAX_NEW_TOKENS" \
            --argjson temperature    0.0 \
            --argjson top_p          0.9 \
            '{messages:$messages,
              max_new_tokens:$max_new_tokens,
              temperature:$temperature,
              top_p:$top_p,
              stop:["\n\n"]}')
        RESP=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '__CURL_FAILED__')
        if is_json "$RESP"; then
            GENERATED=$(json_read "$RESP" '.choices[0].message.content' '')
            PROMPT_T=$( json_read "$RESP" '.usage.prompt_tokens'        '0')
            COMP_T=$(   json_read "$RESP" '.usage.completion_tokens'    '0')
            LATENCY=$(  json_read "$RESP" '.latency_sec'                '0')
            TPS=$(      json_read "$RESP" '.tokens_per_sec'             '0')
        else
            GENERATED=""; PROMPT_T=0; COMP_T=0; LATENCY=0; TPS=0
        fi
    else
        PAYLOAD=$(jq -n \
            --arg     prompt         "$LLM_PROMPT" \
            --argjson max_new_tokens "$EFFECTIVE_MAX_NEW_TOKENS" \
            --argjson temperature    0.0 \
            --argjson top_p          0.9 \
            '{prompt:$prompt,
              max_new_tokens:$max_new_tokens,
              temperature:$temperature,
              top_p:$top_p}')
        RESP=$(curl -sS -X POST "http://localhost:$PORT/generate" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '__CURL_FAILED__')
        if is_json "$RESP"; then
            GENERATED=$(json_read "$RESP" '.generated_text'   '')
            PROMPT_T=$( json_read "$RESP" '.prompt_tokens'    '0')
            COMP_T=$(   json_read "$RESP" '.completion_tokens''0')
            LATENCY=$(  json_read "$RESP" '.latency_sec'      '0')
            TPS=$(      json_read "$RESP" '.tokens_per_sec'   '0')
        else
            GENERATED=""; PROMPT_T=0; COMP_T=0; LATENCY=0; TPS=0
        fi
    fi

    # ── 실패 처리 ─────────────────────────────────────────────────────────────
    if [[ -z "$GENERATED" || "$GENERATED" == "null" ]]; then
        echo "FAIL"
        FAIL=$((FAIL + 1))
        if is_json "$RESP"; then
            ERR=$(json_read "$RESP" '.detail // .error' 'empty response')
        else
            ERR="$RESP"
        fi
        jq -nc \
            --arg ts     "$TS" \
            --arg status "fail" \
            --arg error  "$ERR" \
            '{timestamp:$ts, status:$status, error:$error}' >> "$OUT_FILE"
        continue
    fi

    SUCCESS=$((SUCCESS + 1))

    # ── 검증: LLM 응답에서 severity, root_cause 파싱 ──────────────────────────
    # grep -oi: 대소문자 무시  |  sed: 콜론 이전 제거  |  tr: 하이픈→언더스코어
    LLM_SEV=$(echo "$GENERATED" \
        | grep -oi 'severity[[:space:]]*:[[:space:]]*[a-z]*' \
        | head -1 \
        | sed 's/.*:[[:space:]]*//' \
        | tr '[:upper:]' '[:lower:]')
    LLM_RC=$(echo "$GENERATED" \
        | grep -oi 'root_cause[[:space:]]*:[[:space:]]*[a-z_]*' \
        | head -1 \
        | sed 's/.*:[[:space:]]*//' \
        | tr '[:upper:]' '[:lower:]' \
        | tr '-' '_')

    SEV_OK=$([[ "$LLM_SEV" == "$SEVERITY_RULE" ]] && echo "Y" || echo "N")
    RC_OK=$( [[ "$LLM_RC"  == "$RC_RULE"       ]] && echo "Y" || echo "N")

    [[ "$SEV_OK" == "Y" ]] && SEV_MATCH=$((SEV_MATCH + 1))
    [[ "$RC_OK"  == "Y" ]] && RC_MATCH=$(( RC_MATCH  + 1))

    printf "%6.2fs %6.1f tok/s  sev=%s rc=%s\n" \
        "$LATENCY" "$TPS" "$SEV_OK" "$RC_OK"

    # 누적 통계
    LATENCIES="$LATENCIES $LATENCY"
    THROUGHPUTS="$THROUGHPUTS $TPS"
    PROMPT_TOKS=$((PROMPT_TOKS + PROMPT_T))
    COMP_TOKS=$(( COMP_TOKS   + COMP_T))

    # ── JSONL 저장 ────────────────────────────────────────────────────────────
    jq -nc \
        --arg  ts          "$TS" \
        --arg  status      "ok" \
        --arg  sev_rule    "$SEVERITY_RULE" \
        --arg  rc_rule     "$RC_RULE" \
        --arg  llm_sev     "$LLM_SEV" \
        --arg  llm_rc      "$LLM_RC" \
        --arg  generated   "$GENERATED" \
        --arg  sev_match   "$SEV_OK" \
        --arg  rc_match    "$RC_OK" \
        --argjson prompt_tokens     "$PROMPT_T" \
        --argjson completion_tokens "$COMP_T" \
        --argjson latency_sec       "$LATENCY" \
        --argjson tokens_per_sec    "$TPS" \
        '{timestamp:    $ts,
          status:       $status,
          rule_based:   {severity: $sev_rule,  root_cause: $rc_rule},
          llm_result:   {severity: $llm_sev,   root_cause: $llm_rc},
          validation:   {severity_match: $sev_match, root_cause_match: $rc_match},
          generated:    $generated,
          metrics: {
            prompt_tokens:     $prompt_tokens,
            completion_tokens: $completion_tokens,
            latency_sec:       $latency_sec,
            tokens_per_sec:    $tokens_per_sec
          }}' >> "$OUT_FILE"

done < "$DIAG_JSONL"

# ── 벤치마크 리포트 ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "       TRAFFIC DATASET BENCHMARK REPORT"
echo "════════════════════════════════════════════════"

# VRAM 최종 상태
HEALTH_FINAL=$(curl -s "http://localhost:$PORT/healthz")
MEM_ALLOC_F=$( echo "$HEALTH_FINAL" | jq -r '.cuda_memory_allocated // 0')
MEM_RESV_F=$(  echo "$HEALTH_FINAL" | jq -r '.cuda_memory_reserved  // 0')
MEM_ALLOC_F_MB=$(echo "$MEM_ALLOC_F" | awk '{printf "%.1f", $1/1024/1024}')
MEM_RESV_F_MB=$( echo "$MEM_RESV_F"  | awk '{printf "%.1f", $1/1024/1024}')

# 통계 계산
if [[ $SUCCESS -gt 0 && -n "${LATENCIES// /}" ]]; then
    read -r AVG_LAT MIN_LAT MAX_LAT < <(echo "$LATENCIES" | awk '{
        n=NF; sum=0; min=$1; max=$1
        for(i=1;i<=n;i++){sum+=$i; if($i<min)min=$i; if($i>max)max=$i}
        printf "%.3f %.3f %.3f\n", sum/n, min, max}')
    read -r AVG_TPS MIN_TPS MAX_TPS < <(echo "$THROUGHPUTS" | awk '{
        n=NF; sum=0; min=$1; max=$1
        for(i=1;i<=n;i++){sum+=$i; if($i<min)min=$i; if($i>max)max=$i}
        printf "%.2f %.2f %.2f\n", sum/n, min, max}')
    TOTAL_WALL=$(echo "$LATENCIES" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.1f", s}')
    TOTAL_TOKS=$((PROMPT_TOKS + COMP_TOKS))
else
    AVG_LAT=0; MIN_LAT=0; MAX_LAT=0
    AVG_TPS=0; MIN_TPS=0; MAX_TPS=0
    TOTAL_WALL=0; TOTAL_TOKS=0
fi

echo ""
echo "[ Run Summary ]"
printf "  %-32s %s\n"    "Model:"        "$MODEL_ID_VAL"
printf "  %-32s %s\n"    "API Mode:"     "$API_MODE"
printf "  %-32s %d\n"    "Success:"      "$SUCCESS"
printf "  %-32s %d\n"    "Failed:"       "$FAIL"
printf "  %-32s %d / %d\n" "Written / Target:" "$(wc -l < "$OUT_FILE" 2>/dev/null || echo 0)" "$N_SAMPLES"

echo ""
echo "[ Validation — LLM vs Rule Engine ]"
if [[ $SUCCESS -gt 0 ]]; then
    SEV_PCT=$(echo "$SEV_MATCH $SUCCESS" | awk '{printf "%.1f", 100*$1/$2}')
    RC_PCT=$( echo "$RC_MATCH  $SUCCESS" | awk '{printf "%.1f", 100*$1/$2}')
    printf "  %-32s %d / %d  (%s%%)\n" "Severity Match:"    "$SEV_MATCH" "$SUCCESS" "$SEV_PCT"
    printf "  %-32s %d / %d  (%s%%)\n" "Root Cause Match:"  "$RC_MATCH"  "$SUCCESS" "$RC_PCT"
else
    echo "  (no successful samples)"
fi

echo ""
echo "[ Latency (sec) ]"
printf "  %-32s %.3f\n" "Average:"  "$AVG_LAT"
printf "  %-32s %.3f\n" "Min:"      "$MIN_LAT"
printf "  %-32s %.3f\n" "Max:"      "$MAX_LAT"
printf "  %-32s %.1f\n" "Total (serial sum):" "$TOTAL_WALL"

echo ""
echo "[ Throughput (tok/s) ]"
printf "  %-32s %.2f\n" "Average:"  "$AVG_TPS"
printf "  %-32s %.2f\n" "Min:"      "$MIN_TPS"
printf "  %-32s %.2f\n" "Max:"      "$MAX_TPS"

echo ""
echo "[ Token Usage ]"
printf "  %-32s %d\n"   "Total Prompt Tokens:"     "$PROMPT_TOKS"
printf "  %-32s %d\n"   "Total Completion Tokens:"  "$COMP_TOKS"
printf "  %-32s %d\n"   "Grand Total:"              "$TOTAL_TOKS"

echo ""
echo "[ VRAM (end of run) ]"
printf "  %-32s %s MB / %s MB\n" "Allocated:"  "$MEM_ALLOC_F_MB"  "$MEM_TOTAL_MB"
printf "  %-32s %s MB / %s MB\n" "Reserved:"   "$MEM_RESV_F_MB"   "$MEM_TOTAL_MB"

echo ""
echo "[ Output ]"
printf "  %-32s %s\n" "JSONL saved:" "$OUT_FILE"
printf "  %-32s %s\n" "Log saved:"   "$LOG_FILE"
echo ""
echo "════════════════════════════════════════════════"
