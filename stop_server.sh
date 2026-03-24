#!/bin/bash
# stop_server.sh — 모델별 서버 종료 (또는 전체 종료)
# 사용법:
#   ./stop_server.sh llama     → llama 서버만 종료
#   ./stop_server.sh deepseek  → deepseek 서버만 종료
#   ./stop_server.sh qwen      → qwen 서버만 종료
#   ./stop_server.sh all       → 모든 서버 + 네트워크 종료

set -e

TARGET="${1:-}"

if [[ "$TARGET" != "llama" && "$TARGET" != "deepseek" && "$TARGET" != "qwen" && "$TARGET" != "all" ]]; then
    echo "Usage: $0 <llama|deepseek|qwen|all>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/jetson_slm_stack/docker-compose.yml"

stop_service() {
    local model="$1" service="$2" port="$3"

    echo "--- $model 서버 종료 ($service) ---"
    docker compose -f "$COMPOSE_FILE" rm -sf "$service" 2>/dev/null || true

    # 종료 확인
    if curl -sf "http://localhost:$port/healthz" > /dev/null 2>&1; then
        echo "[WARN] 포트 $port 에 아직 응답이 있습니다. 강제 종료 시도..."
        docker compose -f "$COMPOSE_FILE" kill "$service" 2>/dev/null || true
        docker compose -f "$COMPOSE_FILE" rm -sf "$service" 2>/dev/null || true
    fi
    echo "[OK] $model 서버 종료 완료"
}

if [[ "$TARGET" == "all" ]]; then
    echo "=== 전체 서버 종료 ==="
    docker compose -f "$COMPOSE_FILE" --profile llama --profile deepseek --profile qwen down
    echo "[OK] 모든 서비스 및 네트워크 종료 완료"
else
    case "$TARGET" in
        llama)    stop_service "llama"    "llama32-server"  8000 ;;
        deepseek) stop_service "deepseek" "deepseek-server" 8001 ;;
        qwen)     stop_service "qwen"     "qwen-server"     8002 ;;
    esac
fi

echo ""
echo "현재 실행 중인 컨테이너:"
docker compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "(없음)"
