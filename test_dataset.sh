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
#
# 예시:
#   ./test_dataset.sh llama
#   ./test_dataset.sh deepseek --max 10 --api generate
#   ./test_dataset.sh llama --split val --max 20
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

MAX_SAMPLES=0
API_MODE="chat"
SPLIT="test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max)    MAX_SAMPLES="$2"; shift 2 ;;
        --api)    API_MODE="$2";    shift 2 ;;
        --split)  SPLIT="$2";       shift 2 ;;
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
DATA_FILE="$COMPOSE_DIR/dataset/generated/${SPLIT}.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/test_slm_output"
ENV_FILE="$COMPOSE_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUTPUT_DIR/dataset_eval_${MODEL_TYPE}_${SPLIT}_${TIMESTAMP}.jsonl"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "[ERROR] Dataset file not found: $DATA_FILE"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

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

API_PATH="$([[ "$API_MODE" == "chat" ]] && echo "v1/chat/completions" || echo "generate")"

# ── 서버 상태 확인 ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Dataset Evaluation: $MODEL_TYPE  (port $PORT)"
echo "  Split: $SPLIT   API: /$API_PATH"
echo "  temperature=$TEST_TEMPERATURE  max_new_tokens=$TEST_MAX_NEW_TOKENS  top_p=$TEST_TOP_P"
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

echo ""
echo "  Model : $MODEL_ID_VAL"
echo "  VRAM  : ${MEM_ALLOC_MB} MB / ${MEM_TOTAL_MB} MB"
echo "  Data  : $DATA_FILE"
echo "  Samples: $N_SAMPLES / $TOTAL_LINES"
echo "  Output: $OUT_FILE"
echo ""

# ── 메인 평가 루프 ─────────────────────────────────────────────────────────────
IDX=0
SUCCESS=0
FAIL=0
LATENCIES=""
THROUGHPUTS=""
PROMPT_TOKS=0
COMP_TOKS=0

while IFS= read -r line; do
    [[ $IDX -ge $N_SAMPLES ]] && break

    ID=$(          echo "$line" | jq -r '.id          // "unknown"')
    INSTRUCTION=$( echo "$line" | jq -r '.instruction // ""')
    INPUT=$(       echo "$line" | jq -r '.input       // ""')
    EXPECTED=$(    echo "$line" | jq -r '.output      // ""')

    FULL_PROMPT="$INSTRUCTION"
    if [[ -n "$INPUT" ]]; then
        FULL_PROMPT="$INSTRUCTION

Input: $INPUT"
    fi

    IDX=$((IDX + 1))
    printf "[%3d/%d] %-42s " "$IDX" "$N_SAMPLES" "$ID"

    # API 호출
    if [[ "$API_MODE" == "chat" ]]; then
        PAYLOAD=$(jq -n \
            --arg     content        "$FULL_PROMPT" \
            --argjson max_new_tokens "$TEST_MAX_NEW_TOKENS" \
            --argjson temperature    "$TEST_TEMPERATURE" \
            --argjson top_p          "$TEST_TOP_P" \
            '{messages: [{role: "user", content: $content}],
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p}')
        RESP=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '{}')
        GENERATED=$(echo "$RESP" | jq -r '.choices[0].message.content // ""')
        PROMPT_T=$( echo "$RESP" | jq -r '.usage.prompt_tokens         // 0')
        COMP_T=$(   echo "$RESP" | jq -r '.usage.completion_tokens     // 0')
        LATENCY=$(  echo "$RESP" | jq -r '.latency_sec                 // 0')
        TPS=$(      echo "$RESP" | jq -r '.tokens_per_sec              // 0')
    else
        PAYLOAD=$(jq -n \
            --arg     prompt         "$FULL_PROMPT" \
            --argjson max_new_tokens "$TEST_MAX_NEW_TOKENS" \
            --argjson temperature    "$TEST_TEMPERATURE" \
            --argjson top_p          "$TEST_TOP_P" \
            '{prompt: $prompt,
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p}')
        RESP=$(curl -s -X POST "http://localhost:$PORT/generate" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '{}')
        GENERATED=$(echo "$RESP" | jq -r '.generated_text  // ""')
        PROMPT_T=$( echo "$RESP" | jq -r '.prompt_tokens   // 0')
        COMP_T=$(   echo "$RESP" | jq -r '.completion_tokens // 0')
        LATENCY=$(  echo "$RESP" | jq -r '.latency_sec      // 0')
        TPS=$(      echo "$RESP" | jq -r '.tokens_per_sec   // 0')
    fi

    if [[ -z "$GENERATED" || "$GENERATED" == "null" ]]; then
        echo "FAIL"
        FAIL=$((FAIL + 1))
        jq -nc \
            --arg id     "$ID" \
            --arg status "fail" \
            --arg error  "$(echo "$RESP" | jq -r '.detail // .error // "empty response"')" \
            '{id: $id, status: $status, error: $error}' >> "$OUT_FILE"
        continue
    fi

    SUCCESS=$((SUCCESS + 1))
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
if [[ $SUCCESS -gt 0 && -n "${LATENCIES// /}" ]]; then
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
printf "  %-32s %s\n"    "Max New Tokens:"         "$TEST_MAX_NEW_TOKENS"
printf "  %-32s %d / %d\n" "Samples (success/total):" $SUCCESS $N_SAMPLES
printf "  %-32s %d\n"    "Failed:"                 $FAIL
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
echo ""
echo "════════════════════════════════════════════════"
