#!/bin/bash
# test_dataset.sh — 데이터셋 기반 SLM 평가 + 벤치마크
#
# 사용법:
#   ./test_dataset.sh [llama|deepseek] [옵션]
#
# 옵션:
#   --max  N              : 최대 N건만 평가 (기본: 전체)
#   --api  generate|chat  : 사용할 API 엔드포인트 (기본: chat)
#   --split train|val|test: 평가할 데이터 분할 (기본: test)
#   --resume [FILE]       : 기존 결과 파일에 이어서 재개
#   --log-file FILE       : 장시간 실행 로그 저장 경로 지정
#
# 예시:
#   ./test_dataset.sh llama
#   ./test_dataset.sh deepseek --max 10 --api generate
#   ./test_dataset.sh llama --split val --max 20
#   ./test_dataset.sh llama --split test --api chat --resume test_slm_output/dataset_eval_llama_test_20260324_193144.jsonl
#   ./test_dataset.sh llama --split test --api chat --log-file test_slm_output/dataset_eval_llama_test.log
#
# 주의: 서버가 이미 실행 중이어야 합니다.
#       먼저 ./test_slm.sh llama (또는 deepseek) 으로 기동하세요.

set -uo pipefail

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
MODEL_TYPE="${1:-}"
if [[ "$MODEL_TYPE" != "llama" && "$MODEL_TYPE" != "deepseek" && "$MODEL_TYPE" != "qwen" ]]; then
    echo "Usage: ./test_dataset.sh [llama|deepseek|qwen] [--max N] [--api generate|chat] [--split train|val|test]"
    exit 1
fi
shift

MAX_SAMPLES=10
API_MODE="chat"
SPLIT="test"
RESUME_FILE=""
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max)    MAX_SAMPLES="$2"; shift 2 ;;
        --api)    API_MODE="$2";    shift 2 ;;
        --split)  SPLIT="$2";       shift 2 ;;
        --resume)
            if [[ $# -gt 1 && "$2" != --* ]]; then
                RESUME_FILE="$2"
                shift 2
            else
                RESUME_FILE="__AUTO__"
                shift 1
            fi
            ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$API_MODE" != "generate" && "$API_MODE" != "chat" ]]; then
    echo "[ERROR] --api must be 'generate' or 'chat'"
    exit 1
fi
if [[ "$SPLIT" != "train" && "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "[ERROR] --split must be 'train', 'val', or 'test'"
    exit 1
fi

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/jetson_slm_stack"
DATA_FILE="$COMPOSE_DIR/dataset/prepared/network_slicing_qos/${SPLIT}.jsonl"
PREP_MANIFEST="$COMPOSE_DIR/dataset/prepared/network_slicing_qos/manifest.prep.json"
OUTPUT_DIR="$SCRIPT_DIR/test_slm_output"
ENV_FILE="$COMPOSE_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUTPUT_DIR/dataset_eval_${MODEL_TYPE}_${SPLIT}_${TIMESTAMP}.jsonl"
RUN_LOG_FILE=""

if [[ ! -f "$DATA_FILE" ]]; then
    echo "[ERROR] Dataset file not found: $DATA_FILE"
    if [[ -f "$PREP_MANIFEST" ]]; then
        echo "[INFO] dataset 준비 결과는 존재하지만 요청한 split 파일이 없습니다."
        echo "[INFO] 먼저 dataset/scripts/prepare_network_slicing_dataset.py 를 다시 실행해 주세요."
        echo "[INFO] 준비 메타데이터: $PREP_MANIFEST"
    fi
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ -n "$RESUME_FILE" ]]; then
    if [[ "$RESUME_FILE" == "__AUTO__" ]]; then
        LATEST_MATCH=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "dataset_eval_${MODEL_TYPE}_${SPLIT}_*.jsonl" | sort | tail -1)
        if [[ -z "$LATEST_MATCH" ]]; then
            echo "[ERROR] 재개할 기존 결과 파일을 찾지 못했습니다."
            exit 1
        fi
        RESUME_FILE="$LATEST_MATCH"
    fi

    if [[ ! -f "$RESUME_FILE" ]]; then
        echo "[ERROR] Resume file not found: $RESUME_FILE"
        exit 1
    fi
    OUT_FILE="$RESUME_FILE"
fi

if [[ -z "$LOG_FILE" ]]; then
    RUN_LOG_FILE="${OUT_FILE%.jsonl}.log"
else
    RUN_LOG_FILE="$LOG_FILE"
fi

touch "$RUN_LOG_FILE"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1

# ── 포트 설정 ─────────────────────────────────────────────────────────────────
if [[ "$MODEL_TYPE" == "llama" ]]; then
    PORT=8000
elif [[ "$MODEL_TYPE" == "deepseek" ]]; then
    PORT=8001
else
    PORT=8002
fi

# ── .env에서 파라미터 읽기 ─────────────────────────────────────────────────────
read_env() {
    grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2
}
TEST_MAX_NEW_TOKENS=$(read_env MAX_NEW_TOKENS); TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-128}
TEST_TEMPERATURE=$(read_env TEMPERATURE);       TEST_TEMPERATURE=${TEST_TEMPERATURE:-0.0}
TEST_TOP_P=$(read_env TOP_P);                   TEST_TOP_P=${TEST_TOP_P:-0.9}

is_json() {
    local input="$1"
    jq -e . >/dev/null 2>&1 <<< "$input"
}

json_read() {
    local body="$1"
    local filter="$2"
    local fallback="$3"
    if is_json "$body"; then
        echo "$body" | jq -r --arg fallback "$fallback" "$filter // \$fallback"
    else
        echo "$fallback"
    fi
}

if [[ "$API_MODE" == "chat" ]]; then
    EFFECTIVE_MAX_NEW_TOKENS=256
else
    EFFECTIVE_MAX_NEW_TOKENS=$TEST_MAX_NEW_TOKENS
fi

API_PATH="$([[ "$API_MODE" == "chat" ]] && echo "v1/chat/completions" || echo "generate")"

# ── 서버 상태 확인 ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Dataset Evaluation: $MODEL_TYPE  (port $PORT)"
echo "  Split: $SPLIT   API: /$API_PATH"
echo "  temperature=$TEST_TEMPERATURE  max_new_tokens=$EFFECTIVE_MAX_NEW_TOKENS  top_p=$TEST_TOP_P"
echo "════════════════════════════════════════════════"

if ! curl -sf "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
    echo ""
    echo "[ERROR] 서버가 실행 중이지 않습니다 (port $PORT)"
    echo "  먼저 실행: ./test_slm.sh $MODEL_TYPE"
    exit 1
fi

HEALTH=$(curl -s "http://localhost:$PORT/healthz")
MODEL_ID_VAL=$(echo "$HEALTH" | jq -r '.model_id              // "N/A"')
MAX_IN=$(      echo "$HEALTH" | jq -r '.max_input_tokens       // "N/A"')
MAX_NEW_DEF=$( echo "$HEALTH" | jq -r '.max_new_tokens_default // "N/A"')
MEM_TOTAL=$(   echo "$HEALTH" | jq -r '.cuda_memory_total      // 0')
MEM_ALLOC=$(   echo "$HEALTH" | jq -r '.cuda_memory_allocated  // 0')
MEM_TOTAL_MB=$(echo "$MEM_TOTAL" | awk '{printf "%.1f", $1/1024/1024}')
MEM_ALLOC_MB=$(echo "$MEM_ALLOC" | awk '{printf "%.1f", $1/1024/1024}')

TOTAL_LINES=$(wc -l < "$DATA_FILE")
if [[ "$MAX_SAMPLES" -gt 0 && "$MAX_SAMPLES" -lt "$TOTAL_LINES" ]]; then
    N_SAMPLES=$MAX_SAMPLES
else
    N_SAMPLES=$TOTAL_LINES
fi

RESUME_SKIP=0
if [[ -f "$OUT_FILE" ]]; then
    RESUME_SKIP=$(wc -l < "$OUT_FILE")
fi

if [[ "$RESUME_SKIP" -gt "$N_SAMPLES" ]]; then
    RESUME_SKIP="$N_SAMPLES"
fi

echo ""
echo "  Model : $MODEL_ID_VAL"
echo "  VRAM  : ${MEM_ALLOC_MB} MB / ${MEM_TOTAL_MB} MB"
echo "  Data  : $DATA_FILE"
echo "  Samples: $N_SAMPLES / $TOTAL_LINES"
echo "  Output: $OUT_FILE"
echo "  Log   : $RUN_LOG_FILE"
if [[ "$RESUME_SKIP" -gt 0 ]]; then
    echo "  Resume: enabled (${RESUME_SKIP} samples already written)"
else
    echo "  Resume: disabled"
fi
echo ""

if [[ "$RESUME_SKIP" -ge "$N_SAMPLES" ]]; then
    echo "[INFO] 요청한 샘플 수만큼 이미 처리되었습니다."
    exit 0
fi

# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────
IDX=$RESUME_SKIP
SOURCE_IDX=0
SESSION_SUCCESS=0
SESSION_FAIL=0
LATENCIES=""
THROUGHPUTS=""
PROMPT_TOKS=0
COMP_TOKS=0

while IFS= read -r line; do
    SOURCE_IDX=$((SOURCE_IDX + 1))
    if [[ "$SOURCE_IDX" -le "$RESUME_SKIP" ]]; then
        continue
    fi
    [[ $IDX -ge $N_SAMPLES ]] && break

    # 빈 줄이나 공백 줄은 샘플로 취급하지 않는다.
    if [[ -z "${line//[[:space:]]/}" ]]; then
        echo "[WARN] skip empty source line: $SOURCE_IDX"
        continue
    fi

    # JSONL이 깨진 경우 빈 id/빈 instruction 레코드를 만들지 않고 건너뛴다.
    if ! jq -e . >/dev/null 2>&1 <<< "$line"; then
        echo "[WARN] skip malformed JSON source line: $SOURCE_IDX"
        continue
    fi

    ID=$(          echo "$line" | jq -r '.id          // "unknown"')
    INSTRUCTION=$( echo "$line" | jq -r '.instruction // ""')
    INPUT=$(       echo "$line" | jq -r '.input       // ""')
    EXPECTED=$(    echo "$line" | jq -r '.output      // ""')
    SYSTEM=$(      echo "$line" | jq -r '.system      // ""')

    if [[ -z "$ID" || -z "$INSTRUCTION" ]]; then
        echo "[WARN] skip incomplete sample at source line $SOURCE_IDX (id/instruction missing)"
        continue
    fi

    FULL_PROMPT="$INSTRUCTION"
    if [[ -n "$INPUT" ]]; then
        FULL_PROMPT="$INSTRUCTION

Input: $INPUT"
    fi

    IDX=$((IDX + 1))
    printf "[%s][%3d/%d] %-42s " "$(date +%H:%M:%S)" "$IDX" "$N_SAMPLES" "$ID"

    # API 호출
    if [[ "$API_MODE" == "chat" ]]; then
        # few-shot 2쌍: 새 flags-only instruction 포맷 + 3줄 응답 패턴
        FS_USER1="traffic=eMBB-voice; overload=0; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; packet_loss_abnormal=no; signal_critical=no; hard_breach=yes; stable_allowed=no"
        FS_ASS1="QoS state: critical\nReason: latency exceeds PDB and packet loss exceeds PER for eMBB-voice\nAction: prioritize low-latency scheduling and reduce PRB contention"
        FS_USER2="traffic=URLLC; overload=0; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; packet_loss_abnormal=no; signal_critical=no; hard_breach=no; stable_allowed=yes"
        FS_ASS2="QoS state: stable\nReason: all KPI values are within normal limits for URLLC\nAction: maintain current slice policy and continue KPI monitoring"

        if [[ -n "$SYSTEM" ]]; then
            MSG_JSON=$(jq -n \
                --arg s    "$SYSTEM" \
                --arg fu1  "$FS_USER1" \
                --arg fa1  "$FS_ASS1" \
                --arg fu2  "$FS_USER2" \
                --arg fa2  "$FS_ASS2" \
                --arg u    "$FULL_PROMPT" \
                '[{role:"system",content:$s},
                  {role:"user",content:$fu1},{role:"assistant",content:$fa1},
                  {role:"user",content:$fu2},{role:"assistant",content:$fa2},
                  {role:"user",content:$u}]')
        else
            MSG_JSON=$(jq -n \
                --arg u "$FULL_PROMPT" \
                '[{role:"user",content:$u}]')
        fi

        PAYLOAD=$(jq -n \
            --argjson messages       "$MSG_JSON" \
            --argjson max_new_tokens "$EFFECTIVE_MAX_NEW_TOKENS" \
            --argjson temperature    0.0 \
            --argjson top_p          "$TEST_TOP_P" \
            '{messages: $messages,
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p,
              stop: ["Example ", "\n\n\n"]}')
        RESP=$(curl -sS -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '__CURL_FAILED__')
        if is_json "$RESP"; then
            GENERATED=$(json_read "$RESP" '.choices[0].message.content' '')
            PROMPT_T=$( json_read "$RESP" '.usage.prompt_tokens' '0')
            COMP_T=$(   json_read "$RESP" '.usage.completion_tokens' '0')
            LATENCY=$(  json_read "$RESP" '.latency_sec' '0')
            TPS=$(      json_read "$RESP" '.tokens_per_sec' '0')
        else
            GENERATED=""
            PROMPT_T=0
            COMP_T=0
            LATENCY=0
            TPS=0
        fi
    else
        PAYLOAD=$(jq -n \
            --arg     prompt         "$FULL_PROMPT" \
            --argjson max_new_tokens "$EFFECTIVE_MAX_NEW_TOKENS" \
            --argjson temperature    "$TEST_TEMPERATURE" \
            --argjson top_p          "$TEST_TOP_P" \
            '{prompt: $prompt,
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p}')
        RESP=$(curl -sS -X POST "http://localhost:$PORT/generate" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '__CURL_FAILED__')
        if is_json "$RESP"; then
            GENERATED=$(json_read "$RESP" '.generated_text' '')
            PROMPT_T=$( json_read "$RESP" '.prompt_tokens' '0')
            COMP_T=$(   json_read "$RESP" '.completion_tokens' '0')
            LATENCY=$(  json_read "$RESP" '.latency_sec' '0')
            TPS=$(      json_read "$RESP" '.tokens_per_sec' '0')
        else
            GENERATED=""
            PROMPT_T=0
            COMP_T=0
            LATENCY=0
            TPS=0
        fi
    fi

    if [[ -z "$GENERATED" || "$GENERATED" == "null" ]]; then
        echo "FAIL"
        SESSION_FAIL=$((SESSION_FAIL + 1))
        if is_json "$RESP"; then
            ERROR_MSG=$(json_read "$RESP" '.detail // .error' 'empty response')
        else
            ERROR_MSG="$RESP"
        fi
        jq -nc \
            --arg id     "$ID" \
            --arg status "fail" \
            --arg error  "$ERROR_MSG" \
            '{id: $id, status: $status, error: $error}' >> "$OUT_FILE"
        continue
    fi

    SESSION_SUCCESS=$((SESSION_SUCCESS + 1))
    printf "%6.2fs  %6.1f tok/s  [%4s prompt / %4s completion tok]\n" \
        "$LATENCY" "$TPS" "$PROMPT_T" "$COMP_T"

    # 누적
    LATENCIES="$LATENCIES $LATENCY"
    THROUGHPUTS="$THROUGHPUTS $TPS"
    PROMPT_TOKS=$((PROMPT_TOKS + PROMPT_T))
    COMP_TOKS=$((COMP_TOKS + COMP_T))

    # jsonl 저장
    jq -nc \
        --arg  id          "$ID" \
        --arg  status      "ok" \
        --arg  instruction "$INSTRUCTION" \
        --arg  expected    "$EXPECTED" \
        --arg  generated   "$GENERATED" \
        --argjson prompt_tokens     "$PROMPT_T" \
        --argjson completion_tokens "$COMP_T" \
        --argjson latency_sec       "$LATENCY" \
        --argjson tokens_per_sec    "$TPS" \
        '{id: $id,
          status: $status,
          instruction: $instruction,
          expected: $expected,
          generated: $generated,
          metrics: {
            prompt_tokens:     $prompt_tokens,
            completion_tokens: $completion_tokens,
            latency_sec:       $latency_sec,
            tokens_per_sec:    $tokens_per_sec
          }}' >> "$OUT_FILE"

done < "$DATA_FILE"

# ── 벤치마크 리포트 ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "          DATASET BENCHMARK REPORT"
echo "════════════════════════════════════════════════"

# VRAM 최종 상태
HEALTH_FINAL=$(curl -s "http://localhost:$PORT/healthz")
MEM_ALLOC_F=$(   echo "$HEALTH_FINAL" | jq -r '.cuda_memory_allocated // 0')
MEM_RESV_F=$(    echo "$HEALTH_FINAL" | jq -r '.cuda_memory_reserved  // 0')
MEM_ALLOC_F_MB=$(echo "$MEM_ALLOC_F"  | awk '{printf "%.1f", $1/1024/1024}')
MEM_RESV_F_MB=$( echo "$MEM_RESV_F"   | awk '{printf "%.1f", $1/1024/1024}')

# 통계 계산 (awk: float 안전)
if [[ $SESSION_SUCCESS -gt 0 && -n "${LATENCIES// /}" ]]; then
    read AVG_LAT MIN_LAT MAX_LAT < <(echo "$LATENCIES" | awk '{
        n=NF; sum=0; min=$1; max=$1
        for(i=1;i<=n;i++){
            sum+=$i
            if($i<min) min=$i
            if($i>max) max=$i
        }
        printf "%.3f %.3f %.3f\n", sum/n, min, max
    }')
    read AVG_TPS MIN_TPS MAX_TPS < <(echo "$THROUGHPUTS" | awk '{
        n=NF; sum=0; min=$1; max=$1
        for(i=1;i<=n;i++){
            sum+=$i
            if($i<min) min=$i
            if($i>max) max=$i
        }
        printf "%.2f %.2f %.2f\n", sum/n, min, max
    }')
    TOTAL_TOKS=$((PROMPT_TOKS + COMP_TOKS))
    TOTAL_WALL=$(echo "$LATENCIES" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.1f", s}')
else
    AVG_LAT=0; MIN_LAT=0; MAX_LAT=0
    AVG_TPS=0; MIN_TPS=0; MAX_TPS=0
    TOTAL_TOKS=0; TOTAL_WALL=0
fi

echo ""
echo "[ Run Summary ]"
printf "  %-32s %s\n"    "Model:"                  "$MODEL_ID_VAL"
printf "  %-32s %s\n"    "Split:"                  "$SPLIT"
printf "  %-32s %s\n"    "API Mode:"               "$API_MODE"
printf "  %-32s %s\n"    "Temperature:"            "$TEST_TEMPERATURE"
printf "  %-32s %s\n"    "Max New Tokens:"         "$EFFECTIVE_MAX_NEW_TOKENS"
printf "  %-32s %d\n"    "Previously Completed:"   "$RESUME_SKIP"
printf "  %-32s %d\n"    "Session Success:"        "$SESSION_SUCCESS"
printf "  %-32s %d\n"    "Session Failed:"         "$SESSION_FAIL"
printf "  %-32s %d / %d\n" "Total Written / Target:" "$(wc -l < "$OUT_FILE")" "$N_SAMPLES"
echo ""
echo "[ Latency (sec) ]"
printf "  %-32s %.3f\n"  "Average:"  "$AVG_LAT"
printf "  %-32s %.3f\n"  "Min:"      "$MIN_LAT"
printf "  %-32s %.3f\n"  "Max:"      "$MAX_LAT"
printf "  %-32s %.1f\n"  "Total (serial sum):" "$TOTAL_WALL"
echo ""
echo "[ Throughput (tok/s) ]"
printf "  %-32s %.2f\n"  "Average:"  "$AVG_TPS"
printf "  %-32s %.2f\n"  "Min:"      "$MIN_TPS"
printf "  %-32s %.2f\n"  "Max:"      "$MAX_TPS"
echo ""
echo "[ Token Usage ]"
printf "  %-32s %d\n"    "Total Prompt Tokens:"      $PROMPT_TOKS
printf "  %-32s %d\n"    "Total Completion Tokens:"  $COMP_TOKS
printf "  %-32s %d\n"    "Grand Total:"              $TOTAL_TOKS
echo ""
echo "[ VRAM (end of run) ]"
printf "  %-32s %s MB / %s MB\n"  "Allocated:"  "$MEM_ALLOC_F_MB" "$MEM_TOTAL_MB"
printf "  %-32s %s MB / %s MB\n"  "Reserved:"   "$MEM_RESV_F_MB"  "$MEM_TOTAL_MB"
echo ""
echo "[ Output ]"
printf "  %-32s %s\n"    "JSONL saved:" "$OUT_FILE"
printf "  %-32s %s\n"    "Run log saved:" "$RUN_LOG_FILE"
echo ""
echo "════════════════════════════════════════════════"
