#!/bin/bash
# start_server.sh — 모델별 서버 백그라운드 기동 + 헬스체크 대기
# 사용법:
#   ./start_server.sh llama              → llama 서버 시작
#   ./start_server.sh deepseek           → deepseek 서버 시작
#   ./start_server.sh qwen               → qwen 서버 시작
#   ./start_server.sh llama --rebuild    → 이미지 재빌드 후 시작

set -e

MODEL_TYPE="${1:-}"
REBUILD="${2:-}"

# ── 인자 검증 ──
if [[ "$MODEL_TYPE" != "llama" && "$MODEL_TYPE" != "deepseek" && "$MODEL_TYPE" != "qwen" ]]; then
    echo "Usage: $0 <llama|deepseek|qwen> [--rebuild]"
    exit 1
fi

# ── 모델별 설정 ──
case "$MODEL_TYPE" in
    llama)    PORT=8000; SERVICE="llama32-server"; PROFILE="llama"   ;;
    deepseek) PORT=8001; SERVICE="deepseek-server"; PROFILE="deepseek" ;;
    qwen)     PORT=8002; SERVICE="qwen-server";     PROFILE="qwen"   ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/jetson_slm_stack/docker-compose.yml"

# ── 이미 실행 중인지 확인 ──
if curl -sf "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
    echo "[INFO] $MODEL_TYPE 서버가 이미 포트 $PORT 에서 실행 중입니다."
    curl -s "http://localhost:$PORT/healthz" | python3 -m json.tool
    exit 0
fi

# ── 기존 컨테이너 정리 ──
echo "--- [1/3] 기존 $SERVICE 컨테이너 정리 ---"
docker compose -f "$COMPOSE_FILE" rm -sf "$SERVICE" 2>/dev/null || true

# ── 빌드 ──
echo "--- [2/3] Docker 이미지 빌드 ---"
if [[ "$REBUILD" == "--rebuild" ]]; then
    docker compose -f "$COMPOSE_FILE" build --no-cache "$SERVICE"
else
    docker compose -f "$COMPOSE_FILE" build "$SERVICE"
fi

# ── 백그라운드 서버 시작 ──
echo "--- [3/3] $MODEL_TYPE 서버 시작 (포트 $PORT) ---"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d "$SERVICE"

# ── 헬스체크 대기 (최대 150초) ──
MAX_WAIT=30  # 30회 × 5초 = 150초
echo "서버 준비 대기 중 (최대 $((MAX_WAIT * 5))초)..."
for i in $(seq 1 $MAX_WAIT); do
    sleep 5
    if curl -sf "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
        echo ""
        echo "=== $MODEL_TYPE 서버 준비 완료! (${i}×5 = $((i * 5))초) ==="
        curl -s "http://localhost:$PORT/healthz" | python3 -m json.tool
        exit 0
    fi
    echo "  대기 중... ($((i * 5))초)"
done

echo ""
echo "[ERROR] $MODEL_TYPE 서버가 $((MAX_WAIT * 5))초 내에 시작되지 않았습니다."
echo "로그 확인: docker compose -f $COMPOSE_FILE logs $SERVICE"
exit 1
