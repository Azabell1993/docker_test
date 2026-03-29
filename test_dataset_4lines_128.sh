#!/bin/bash
# test_dataset_4lines_128.sh — 4개 샘플만 별도로 128 토큰 설정으로 평가

set -uo pipefail

MODEL_TYPE="${1:-}"
if [[ "$MODEL_TYPE" != "llama" && "$MODEL_TYPE" != "deepseek" && "$MODEL_TYPE" != "qwen" ]]; then
    echo "Usage: ./test_dataset_4lines_128.sh [llama|deepseek|qwen] [--api generate|chat] [--split train|val|test]"
    exit 1
fi
shift

API_MODE="chat"
SPLIT="test"
SAMPLE_COUNT=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api)   API_MODE="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/jetson_slm_stack"
DATA_FILE="$COMPOSE_DIR/dataset/prepared/network_slicing_qos/${SPLIT}.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/test_slm_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUTPUT_DIR/dataset_eval_${MODEL_TYPE}_${SPLIT}_4samples_128_${TIMESTAMP}.jsonl"
LOG_FILE="${OUT_FILE%.jsonl}.log"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "[ERROR] Dataset file not found: $DATA_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$MODEL_TYPE" == "llama" ]]; then
    PORT=8000
elif [[ "$MODEL_TYPE" == "deepseek" ]]; then
    PORT=8001
else
    PORT=8002
fi

TEST_MAX_NEW_TOKENS=128
TEST_TEMPERATURE=0.0
TEST_TOP_P=1.0
API_PATH="$([[ "$API_MODE" == "chat" ]] && echo "v1/chat/completions" || echo "generate")"

echo ""
echo "════════════════════════════════════════════════"
echo "  4-Sample Evaluation: $MODEL_TYPE  (port $PORT)"
echo "  Split: $SPLIT   API: /$API_PATH"
echo "  temperature=$TEST_TEMPERATURE  max_new_tokens=$TEST_MAX_NEW_TOKENS  top_p=$TEST_TOP_P"
echo "════════════════════════════════════════════════"

if ! curl -sf "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
    echo ""
    echo "[ERROR] 서버가 실행 중이지 않습니다 (port $PORT)"
    echo "  먼저 실행: ./start_server.sh $MODEL_TYPE"
    exit 1
fi

echo ""
echo "  Data  : $DATA_FILE"
echo "  Limit : first $SAMPLE_COUNT rows"
echo "  Output: $OUT_FILE"
echo "  Log   : $LOG_FILE"
echo ""

IDX=0

while IFS= read -r line; do
    [[ $IDX -ge $SAMPLE_COUNT ]] && break
    IDX=$((IDX + 1))

    ID=$(echo "$line" | jq -r '.id // "unknown"')
    INSTRUCTION=$(echo "$line" | jq -r '.instruction // ""')
    INPUT=$(echo "$line" | jq -r '.input // ""')
    EXPECTED=$(echo "$line" | jq -r '.output // ""')

    FULL_PROMPT="$INSTRUCTION"
    if [[ -n "$INPUT" ]]; then
        FULL_PROMPT="$INSTRUCTION

Input: $INPUT"
    fi

    printf "[%d/%d] %-42s " "$IDX" "$SAMPLE_COUNT" "$ID"

    if [[ "$API_MODE" == "chat" ]]; then
        PAYLOAD=$(jq -n \
            --arg content "$FULL_PROMPT" \
            --argjson max_new_tokens "$TEST_MAX_NEW_TOKENS" \
            --argjson temperature "$TEST_TEMPERATURE" \
            --argjson top_p "$TEST_TOP_P" \
            '{messages: [{role: "user", content: $content}],
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p}')
        RESP=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '{}')
        GENERATED=$(echo "$RESP" | jq -r '.choices[0].message.content // ""')
        PROMPT_T=$(echo "$RESP" | jq -r '.usage.prompt_tokens // 0')
        COMP_T=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
        LATENCY=$(echo "$RESP" | jq -r '.latency_sec // 0')
        TPS=$(echo "$RESP" | jq -r '.tokens_per_sec // 0')
    else
        PAYLOAD=$(jq -n \
            --arg prompt "$FULL_PROMPT" \
            --argjson max_new_tokens "$TEST_MAX_NEW_TOKENS" \
            --argjson temperature "$TEST_TEMPERATURE" \
            --argjson top_p "$TEST_TOP_P" \
            '{prompt: $prompt,
              max_new_tokens: $max_new_tokens,
              temperature: $temperature,
              top_p: $top_p}')
        RESP=$(curl -s -X POST "http://localhost:$PORT/generate" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" || echo '{}')
        GENERATED=$(echo "$RESP" | jq -r '.generated_text // ""')
        PROMPT_T=$(echo "$RESP" | jq -r '.prompt_tokens // 0')
        COMP_T=$(echo "$RESP" | jq -r '.completion_tokens // 0')
        LATENCY=$(echo "$RESP" | jq -r '.latency_sec // 0')
        TPS=$(echo "$RESP" | jq -r '.tokens_per_sec // 0')
    fi

    if [[ -z "$GENERATED" || "$GENERATED" == "null" ]]; then
        echo "FAIL"
        jq -nc \
            --arg id "$ID" \
            --arg status "fail" \
            --arg expected "$EXPECTED" \
            --arg error "$(echo "$RESP" | jq -r '.detail // .error // "empty response"')" \
            '{id: $id, status: $status, expected: $expected, error: $error}' >> "$OUT_FILE"
        continue
    fi

    printf "%6.2fs  %6.1f tok/s  [%4s prompt / %4s completion tok]\n" \
        "$LATENCY" "$TPS" "$PROMPT_T" "$COMP_T"

    jq -nc \
        --arg id "$ID" \
        --arg status "ok" \
        --arg instruction "$INSTRUCTION" \
        --arg expected "$EXPECTED" \
        --arg generated "$GENERATED" \
        --argjson prompt_tokens "$PROMPT_T" \
        --argjson completion_tokens "$COMP_T" \
        --argjson latency_sec "$LATENCY" \
        --argjson tokens_per_sec "$TPS" \
        '{id: $id,
          status: $status,
          instruction: $instruction,
          expected: $expected,
          generated: $generated,
          metrics: {
            prompt_tokens: $prompt_tokens,
            completion_tokens: $completion_tokens,
            latency_sec: $latency_sec,
            tokens_per_sec: $tokens_per_sec
          }}' >> "$OUT_FILE"
done < "$DATA_FILE"

echo ""
echo "════════════════════════════════════════════════"
echo "  4-sample test completed"
echo "════════════════════════════════════════════════"
echo "  JSONL saved: $OUT_FILE"
echo "  Run log saved: $LOG_FILE"
