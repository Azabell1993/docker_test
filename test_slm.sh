#!/bin/bash

# 사용법: ./test_slm.sh [llama|deepseek] [--rebuild]
#   --rebuild : 강제로 이미지 재빌드 후 서버 재시작
MODEL_TYPE=$1
REBUILD=${2:-""}

if [ "$MODEL_TYPE" != "llama" ] && [ "$MODEL_TYPE" != "deepseek" ] && [ "$MODEL_TYPE" != "qwen" ]; then
    echo "Usage: ./test_slm.sh [llama|deepseek|qwen] [--rebuild]"
    exit 1
fi

# 포트 및 프롬프트 설정
if [ "$MODEL_TYPE" == "llama" ]; then
    PORT=8000
    DEFAULT_PROMPT="What is the capital of France?"
    SERVICE_NAME="llama32-server"
elif [ "$MODEL_TYPE" == "deepseek" ]; then
    PORT=8001
    DEFAULT_PROMPT="Explain underwater acoustic communication simply."
    SERVICE_NAME="deepseek-server"
else
    PORT=8002
    DEFAULT_PROMPT="Explain underwater acoustic communication simply."
    SERVICE_NAME="qwen-server"
fi

# 프롬프트 입력 — 파이프(stdin) 또는 인터랙티브 모두 지원
#   echo 'Hello' | ./test_slm.sh qwen   → 파이프에서 읽음
#   ./test_slm.sh qwen                  → 터미널에서 직접 입력
echo ""
if [ ! -t 0 ]; then
    # stdin이 파이프/리다이렉트인 경우
    read -r USER_PROMPT
else
    # 터미널에서 직접 실행한 경우
    read -r -p "Enter prompt [default: $DEFAULT_PROMPT]: " USER_PROMPT
fi
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

is_json() {
    local input="$1"
    jq -e . >/dev/null 2>&1 <<< "$input"
}

print_json_or_raw() {
    local body="$1"
    if is_json "$body"; then
        echo "$body" | jq
    else
        printf '%s\n' "$body"
    fi
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

request_json() {
    local method="$1"
    local url="$2"
    local payload="${3:-}"
    local response

    if [ -n "$payload" ]; then
        response=$(curl -sS -X "$method" "$url" -H "Content-Type: application/json" -d "$payload" -w $'\n%{http_code}')
    else
        response=$(curl -sS -X "$method" "$url" -w $'\n%{http_code}')
    fi

    HTTP_STATUS="${response##*$'\n'}"
    HTTP_BODY="${response%$'\n'*}"
}

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
request_json GET "http://localhost:$PORT/healthz"
HEALTH_STATUS="$HTTP_STATUS"
HEALTH="$HTTP_BODY"
print_json_or_raw "$HEALTH"
echo -e "\n"

echo "--- [TEST] Models List ---"
request_json GET "http://localhost:$PORT/v1/models"
MODELS_STATUS="$HTTP_STATUS"
MODELS_BODY="$HTTP_BODY"
print_json_or_raw "$MODELS_BODY"
echo -e "\n"

echo "--- [TEST] Generation API ---"
echo "  Prompt: \"$PROMPT\""
GEN_PAYLOAD=$(jq -n \
  --arg     prompt         "$PROMPT" \
  --argjson max_new_tokens $TEST_MAX_NEW_TOKENS \
  --argjson temperature    $TEST_TEMPERATURE \
  --argjson top_p          $TEST_TOP_P \
  '{prompt: $prompt, max_new_tokens: $max_new_tokens, temperature: $temperature, top_p: $top_p}')
request_json POST "http://localhost:$PORT/generate" "$GEN_PAYLOAD"
GEN_STATUS="$HTTP_STATUS"
GEN_RESP="$HTTP_BODY"
if [ "$GEN_STATUS" != "200" ]; then
        echo "  [HTTP $GEN_STATUS]"
fi
print_json_or_raw "$GEN_RESP"
echo -e "\n"

echo "--- [TEST] Chat Completions API ---"
echo "  Prompt: \"$PROMPT\""
CHAT_PAYLOAD=$(jq -n \
  --arg     content        "$PROMPT" \
  --argjson max_new_tokens $TEST_MAX_NEW_TOKENS \
  --argjson temperature    $TEST_TEMPERATURE \
  --argjson top_p          $TEST_TOP_P \
  '{messages: [{role: "user", content: $content}], max_new_tokens: $max_new_tokens, temperature: $temperature, top_p: $top_p}')
request_json POST "http://localhost:$PORT/v1/chat/completions" "$CHAT_PAYLOAD"
CHAT_STATUS="$HTTP_STATUS"
CHAT_RESP="$HTTP_BODY"
if [ "$CHAT_STATUS" != "200" ]; then
        echo "  [HTTP $CHAT_STATUS]"
fi
print_json_or_raw "$CHAT_RESP"
echo -e "\n"

# ──────────────────────────────────────────────
# PERFORMANCE REPORT
# ──────────────────────────────────────────────
echo "════════════════════════════════════════════════"
echo "             PERFORMANCE REPORT                 "
echo "════════════════════════════════════════════════"

# healthz 필드
MODEL_ID_VAL=$(json_read "$HEALTH" '.model_id' 'N/A')
MODEL_SRC=$(json_read "$HEALTH" '.model_source' 'N/A')
CUDA_ACTIVE=$(json_read "$HEALTH" '.active_device' 'N/A')
CUDA_AVAIL=$(json_read "$HEALTH" '.cuda_available' 'N/A')
DTYPE=$(json_read "$HEALTH" '.dtype' 'N/A')
MAX_IN=$(json_read "$HEALTH" '.max_input_tokens' '0')
MAX_NEW_DEF=$(json_read "$HEALTH" '.max_new_tokens_default' '0')
MEM_ALLOC=$(json_read "$HEALTH" '.cuda_memory_allocated' '0')
MEM_RESV=$(json_read "$HEALTH" '.cuda_memory_reserved' '0')

# Generation API 필드
GEN_PROMPT_TOK=$(json_read "$GEN_RESP" '.prompt_tokens' 'N/A')
GEN_COMP_TOK=$(json_read "$GEN_RESP" '.completion_tokens' 'N/A')
GEN_TOTAL_TOK=$(json_read "$GEN_RESP" '.total_tokens' 'N/A')
GEN_LATENCY=$(json_read "$GEN_RESP" '.latency_sec' 'N/A')
GEN_TPS=$(json_read "$GEN_RESP" '.tokens_per_sec' 'N/A')

# Chat Completions API 필드
CHAT_PROMPT_TOK=$(json_read "$CHAT_RESP" '.usage.prompt_tokens' 'N/A')
CHAT_COMP_TOK=$(json_read "$CHAT_RESP" '.usage.completion_tokens' 'N/A')
CHAT_TOTAL_TOK=$(json_read "$CHAT_RESP" '.usage.total_tokens' 'N/A')
CHAT_LATENCY=$(json_read "$CHAT_RESP" '.latency_sec' 'N/A')
CHAT_TPS=$(json_read "$CHAT_RESP" '.tokens_per_sec' 'N/A')

# 메모리 MB 변환
MEM_TOTAL=$(json_read "$HEALTH" '.cuda_memory_total' '0')

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