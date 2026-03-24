#!/bin/bash

set -e

CMD=$1

cd ~/jiwoo_research_docker/jetson_slm_stack

case "$CMD" in
  prep)
    echo "[STEP] Dataset Preparation"
    docker compose --profile "" up --build dataset-prep
    ;;

  download)
    echo "[STEP] Model Download"
    docker compose --profile "" up --build model-download
    ;;

  llama)
    echo "[STEP] Start LLaMA Server"
    docker compose --profile llama up --build -d llama32-server
    ;;

  deepseek)
    echo "[STEP] Start DeepSeek Server"
    docker compose --profile deepseek up --build -d deepseek-server
    ;;

  qwen)
    echo "[STEP] Start Qwen Server"
    docker compose --profile qwen up --build -d qwen-server
    ;;

  clean)
    echo "[STEP] Clean Docker"
    docker compose down
    docker builder prune -af
    ;;

  *)
    echo "Usage: $0 {prep|download|llama|deepseek|qwen|clean}"
    exit 1
    ;;
esac