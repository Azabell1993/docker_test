#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/release"
mkdir -p "$OUT_DIR"

cp "$ROOT_DIR/.env.example" "$OUT_DIR/.env.dgx.example"
sed -i 's#DOCKERFILE_PATH=docker/Dockerfile.jetson#DOCKERFILE_PATH=docker/Dockerfile.dgx#' "$OUT_DIR/.env.dgx.example"
sed -i 's#BASE_IMAGE=.*#BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04#' "$OUT_DIR/.env.dgx.example"

tar -czf "$OUT_DIR/marine-dx-slm-stack-dgx-portable.tar.gz" \
  -C "$ROOT_DIR" \
  app dataset docker docker-compose.yml scripts .env.example README.md

echo "Created: $OUT_DIR/marine-dx-slm-stack-dgx-portable.tar.gz"
