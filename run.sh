#!/bin/bash
# run.sh — 전체 스택 초기화 + 서버 기동
# 사용법: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/jetson_slm_stack"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
RUN_JETSON="$COMPOSE_DIR/scripts/run_jetson.sh"

cd "$COMPOSE_DIR"

echo "--- [1/4] Stopping all services ---"
docker compose -f "$COMPOSE_FILE" down

echo "--- [2/4] Pruning build cache ---"
docker builder prune -af

echo "--- [3/4] Building images ---"
docker compose -f "$COMPOSE_FILE" build --no-cache

echo "--- [4/4] Starting llama server ---"
"$RUN_JETSON" llama
