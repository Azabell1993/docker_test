#!/bin/bash

# 사용법: ./test_slm.sh [llama|deepseek] [--rebuild]
#   --rebuild : 강제로 이미지 재빌드 후 서버 재시작
MODEL_TYPE=$1
REBUILD=${2:-""}

if [ "$MODEL_TYPE" != "llama" ] && [ "$MODEL_TYPE" != "deepseek" ]; then
    echo "Usage: ./test_slm.sh [llama|deepseek] [--rebuild]"
    exit 1
fi

# 포트 및 프롬프트 설정
if [ "$MODEL_TYPE" == "llama" ]; then
    PORT=8000
    DEFAULT_PROMPT="What is the capital of France?"
    SERVICE_NAME="llama32-server"
else
    PORT=8001
    DEFAULT_PROMPT="Explain underwater acoustic communication simply."
    SERVICE_NAME="deepseek-server"
fi

# 프롬프트 직접 입력 (엔터만 누르면 기본값 사용)
echo ""
read -r -p "Enter prompt [default: $DEFAULT_PROMPT]: " USER_PROMPT
PROMPT="${USER_PROMPT:-$DEFAULT_PROMPT}"
echo "  Using prompt: \"$PROMPT\""
echo ""

COMPOSE_DIR="$(cd "$(dirname "$0")/jetson_slm_stack" && pwd)"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

# .env에서 MAX_NEW_TOKENS 읽기 (없으면 128 기본값)
ENV_FILE="$COMPOSE_DIR/.env"
TEST_MAX_NEW_TOKENS=$(grep -E '^MAX_NEW_TOKENS=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2)
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-128}
TEST_TEMPERATURE=$(grep -E '^TEMPERATURE=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2)
TEST_TEMPERATURE=${TEST_TEMPERATURE:-0.2}
TEST_TOP_P=$(grep -E '^TOP_P=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2)
TEST_TOP_P=${TEST_TOP_P:-0.9}

echo "  [Config] max_new_tokens=$TEST_MAX_NEW_TOKENS  temperature=$TEST_TEMPERATURE  top_p=$TEST_TOP_P"
echo ""

# 서버가 이미 올라와 있고 --rebuild 아니면 테스트만 실행
if curl -sf http://localhost:$PORT/healthz > /dev/null 2>&1 && [ "$REBUILD" != "--rebuild" ]; then
    echo "--- Server already running on port $PORT, skipping startup ---"
else
    echo "--- [1/3] Stopping $SERVICE_NAME (other services untouched) ---"
    docker compose -f "$COMPOSE_FILE" rm -sf $SERVICE_NAME

    echo "--- [2/3] Building Docker Image ---"
    if [ "$REBUILD" == "--rebuild" ]; then
        docker compose -f "$COMPOSE_FILE" build --no-cache $SERVICE_NAME
    else
        docker compose -f "$COMPOSE_FILE" build $SERVICE_NAME
    fi

    echo "--- [3/3] Starting $MODEL_TYPE Server ---"
    "$COMPOSE_DIR/scripts/run_jetson.sh" $MODEL_TYPE &

    echo "Waiting for server to be ready (max 120s)..."
    for i in $(seq 1 24); do
        sleep 5
        if curl -sf http://localhost:$PORT/healthz > /dev/null 2>&1; then
            echo "Server is up!"
            break
        fi
        echo "  Still waiting... ($((i*5))s)"
    done
fi

echo "--- [TEST] Health Check ---"
HEALTH=$(curl -s http://localhost:$PORT/healthz)
echo "$HEALTH" | jq
echo -e "\n"

echo "--- [TEST] Models List ---"
curl -s http://localhost:$PORT/v1/models | jq
echo -e "\n"

echo "--- [TEST] Generation API ---"
echo "  Prompt: \"$PROMPT\""
GEN_PAYLOAD=$(jq -n \
  --arg     prompt         "$PROMPT" \
  --argjson max_new_tokens $TEST_MAX_NEW_TOKENS \
  --argjson temperature    $TEST_TEMPERATURE \
  --argjson top_p          $TEST_TOP_P \
  '{prompt: $prompt, max_new_tokens: $max_new_tokens, temperature: $temperature, top_p: $top_p}')
GEN_RESP=$(curl -s -X POST http://localhost:$PORT/generate \
  -H "Content-Type: application/json" \
  -d "$GEN_PAYLOAD")
echo "$GEN_RESP" | jq
echo -e "\n"

echo "--- [TEST] Chat Completions API ---"
echo "  Prompt: \"$PROMPT\""
CHAT_PAYLOAD=$(jq -n \
  --arg     content        "$PROMPT" \
  --argjson max_new_tokens $TEST_MAX_NEW_TOKENS \
  --argjson temperature    $TEST_TEMPERATURE \
  --argjson top_p          $TEST_TOP_P \
  '{messages: [{role: "user", content: $content}], max_new_tokens: $max_new_tokens, temperature: $temperature, top_p: $top_p}')
CHAT_RESP=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$CHAT_PAYLOAD")
echo "$CHAT_RESP" | jq
echo -e "\n"

# ──────────────────────────────────────────────
# PERFORMANCE REPORT
# ──────────────────────────────────────────────
echo "════════════════════════════════════════════════"
echo "             PERFORMANCE REPORT                 "
echo "════════════════════════════════════════════════"

# healthz 필드
MODEL_ID_VAL=$(echo "$HEALTH"   | jq -r '.model_id          // "N/A"')
MODEL_SRC=$(echo "$HEALTH"      | jq -r '.model_source       // "N/A"')
CUDA_ACTIVE=$(echo "$HEALTH"    | jq -r '.active_device      // "N/A"')
CUDA_AVAIL=$(echo "$HEALTH"     | jq -r '.cuda_available     // "N/A"')
DTYPE=$(echo "$HEALTH"          | jq -r '.dtype              // "N/A"')
MAX_IN=$(echo "$HEALTH"         | jq -r '.max_input_tokens        // "N/A"')
MAX_NEW_DEF=$(echo "$HEALTH"    | jq -r '.max_new_tokens_default   // "N/A"')
MEM_ALLOC=$(echo "$HEALTH"      | jq -r '.cuda_memory_allocated   // 0')
MEM_RESV=$(echo "$HEALTH"       | jq -r '.cuda_memory_reserved    // 0')

# Generation API 필드
GEN_PROMPT_TOK=$(echo "$GEN_RESP"  | jq -r '.prompt_tokens     // "N/A"')
GEN_COMP_TOK=$(echo "$GEN_RESP"    | jq -r '.completion_tokens // "N/A"')
GEN_TOTAL_TOK=$(echo "$GEN_RESP"   | jq -r '.total_tokens      // "N/A"')
GEN_LATENCY=$(echo "$GEN_RESP"     | jq -r '.latency_sec       // "N/A"')
GEN_TPS=$(echo "$GEN_RESP"         | jq -r '.tokens_per_sec    // "N/A"')

# Chat Completions API 필드
CHAT_PROMPT_TOK=$(echo "$CHAT_RESP" | jq -r '.usage.prompt_tokens     // "N/A"')
CHAT_COMP_TOK=$(echo "$CHAT_RESP"   | jq -r '.usage.completion_tokens // "N/A"')
CHAT_TOTAL_TOK=$(echo "$CHAT_RESP"  | jq -r '.usage.total_tokens      // "N/A"')
CHAT_LATENCY=$(echo "$CHAT_RESP"    | jq -r '.latency_sec             // "N/A"')
CHAT_TPS=$(echo "$CHAT_RESP"        | jq -r '.tokens_per_sec          // "N/A"')

# 메모리 MB 변환
MEM_TOTAL=$(echo "$HEALTH" | jq -r '.cuda_memory_total // 0')

MEM_ALLOC_MB=$(echo "$MEM_ALLOC" | awk '{printf "%.1f", $1/1024/1024}')
MEM_RESV_MB=$(echo  "$MEM_RESV"  | awk '{printf "%.1f", $1/1024/1024}')
MEM_TOTAL_MB=$(echo "$MEM_TOTAL" | awk '{printf "%.1f", $1/1024/1024}')

# 퍼센트 계산 헬퍼
pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if(b>0) printf "%.1f", a/b*100; else print "N/A" }'; }

ALLOC_PCT=$(pct "$MEM_ALLOC" "$MEM_TOTAL")
RESV_PCT=$(pct  "$MEM_RESV"  "$MEM_TOTAL")

GEN_PROMPT_PCT=$(pct  "$GEN_PROMPT_TOK" "$MAX_IN")
GEN_COMP_PCT=$(pct   "$GEN_COMP_TOK"   "$MAX_NEW_DEF")
GEN_TOTAL_MAX=$(( MAX_IN + MAX_NEW_DEF ))
GEN_TOTAL_PCT=$(pct  "$GEN_TOTAL_TOK"  "$GEN_TOTAL_MAX")

CHAT_PROMPT_PCT=$(pct "$CHAT_PROMPT_TOK" "$MAX_IN")
CHAT_COMP_PCT=$(pct   "$CHAT_COMP_TOK"   "$MAX_NEW_DEF")
CHAT_TOTAL_PCT=$(pct  "$CHAT_TOTAL_TOK"  "$GEN_TOTAL_MAX")

echo ""
echo "[ Model ]"
printf "  %-30s %s\n" "Model ID:"          "$MODEL_ID_VAL"
printf "  %-30s %s\n" "Model Source:"      "$MODEL_SRC"
printf "  %-30s %s\n" "Active Device:"     "$CUDA_ACTIVE"
printf "  %-30s %s\n" "CUDA Available:"    "$CUDA_AVAIL"
printf "  %-30s %s\n" "dtype:"             "$DTYPE"
echo ""
echo "[ Token Limits ]"
printf "  %-30s %s tokens\n"         "Max Input Tokens:"         "$MAX_IN"
printf "  %-30s %s tokens\n"         "Max New Tokens (default):" "$MAX_NEW_DEF"
printf "  %-30s %s tokens\n"         "Max Total (in+new):"       "$GEN_TOTAL_MAX"
echo ""
echo "[ CUDA Memory  (used / total) ]"
printf "  %-30s %s MB / %s MB  (%s%%)\n" "Allocated:" "$MEM_ALLOC_MB" "$MEM_TOTAL_MB" "$ALLOC_PCT"
printf "  %-30s %s MB / %s MB  (%s%%)\n" "Reserved:"  "$MEM_RESV_MB"  "$MEM_TOTAL_MB" "$RESV_PCT"
echo ""
echo "[ /generate API ]"
printf "  %-30s %s / %s  (%s%%)\n" "Prompt Tokens:"     "$GEN_PROMPT_TOK" "$MAX_IN"      "$GEN_PROMPT_PCT"
printf "  %-30s %s / %s  (%s%%)\n" "Completion Tokens:" "$GEN_COMP_TOK"  "$MAX_NEW_DEF" "$GEN_COMP_PCT"
printf "  %-30s %s / %s  (%s%%)\n" "Total Tokens:"      "$GEN_TOTAL_TOK" "$GEN_TOTAL_MAX" "$GEN_TOTAL_PCT"
printf "  %-30s %s sec\n"           "Latency:"           "$GEN_LATENCY"
printf "  %-30s %s tok/s\n"         "Throughput:"        "$GEN_TPS"
echo ""
echo "[ /v1/chat/completions API ]"
printf "  %-30s %s / %s  (%s%%)\n" "Prompt Tokens:"     "$CHAT_PROMPT_TOK" "$MAX_IN"        "$CHAT_PROMPT_PCT"
printf "  %-30s %s / %s  (%s%%)\n" "Completion Tokens:" "$CHAT_COMP_TOK"   "$MAX_NEW_DEF"   "$CHAT_COMP_PCT"
printf "  %-30s %s / %s  (%s%%)\n" "Total Tokens:"      "$CHAT_TOTAL_TOK"  "$GEN_TOTAL_MAX" "$CHAT_TOTAL_PCT"
printf "  %-30s %s sec\n"           "Latency:"           "$CHAT_LATENCY"
printf "  %-30s %s tok/s\n"         "Throughput:"        "$CHAT_TPS"
echo ""
echo "════════════════════════════════════════════════"