#!/bin/bash
# rerun.sh — llama32-server 재시작 + 메모리 정리
# 사용법:
#   ./rerun.sh           → 빠른 재시작 (이미지 재사용)
#   ./rerun.sh --rebuild → 이미지 완전 재빌드 후 재시작

COMPOSE_FILE="jetson_slm_stack/docker-compose.yml"
REBUILD=${1:-""}

# 1. 서버 컨테이너만 중지 (다른 서비스 유지)
echo "--- [1/4] Stopping llama32-server ---"
docker compose -f "$COMPOSE_FILE" rm -sf llama32-server

# 2. 이미지 재빌드 (--rebuild 옵션 시)
if [ "$REBUILD" = "--rebuild" ]; then
    echo "--- [2/4] Rebuilding image (no-cache) ---"
    docker compose -f "$COMPOSE_FILE" build --no-cache llama32-server
else
    echo "--- [2/4] Skipping rebuild (use --rebuild to force)"
fi

# 3. 메모리 정리
echo "--- [3/4] Cleaning memory ---"
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo swapoff -a && sudo swapon -a
echo "Waiting for memory to settle..."
sleep 3
free -h

# 사용 가능 메모리 확인 (2800MB 이상 필요)
AVAIL_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
if [[ $AVAIL_MB -lt 2800 ]]; then
    echo "[ERROR] 사용 가능 메모리 부족: ${AVAIL_MB}MB (최소 2800MB 필요)"
    echo "  → VSCode, 브라우저 등 다른 프로세스를 종료 후 재시도하세요."
    exit 1
fi
echo "[OK] 사용 가능 메모리: ${AVAIL_MB}MB"

# 4. 서버 시작
echo "--- [4/4] Starting llama32-server ---"
./jetson_slm_stack/scripts/run_jetson.sh llama