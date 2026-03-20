# Marine DX Physical AI + Network SLM Stack for Jetson Orin Nano

This package is a deployable starter stack for:
- Jetson Orin Nano edge inference with Docker Compose
- two SLM candidates: **Meta Llama 3.2 1B Instruct** and **DeepSeek-R1-Distill-Qwen-1.5B**
- a synthetic **marine DX Physical AI + Network** instruction dataset
- a migration path that can be repackaged for **DGX Spark / x86 CUDA** later

## Directory layout

- `docker-compose.yml` : compose entrypoint
- `docker/Dockerfile.jetson` : Jetson-target image
- `docker/Dockerfile.dgx` : x86 CUDA / DGX-target image
- `app/server.py` : FastAPI inference service
- `app/download_models.py` : Hugging Face model downloader
- `dataset/scripts/generate_marine_dx_dataset.py` : synthetic dataset generator
- `scripts/run_jetson.sh` : convenience launcher
- `scripts/package_for_dgx.sh` : DGX repackaging helper

## Assumptions

1. Jetson is already flashed with a JetPack 6.x image.
2. Docker and Compose are available.
3. NVIDIA container runtime is configured on the device.
4. You have a Hugging Face token in order to download both models. For Llama, you must also accept the Meta license on Hugging Face.

## Quick start

```bash
cp .env.example .env
# edit BASE_IMAGE and HF_TOKEN

./scripts/run_jetson.sh prep
./scripts/run_jetson.sh download
./scripts/run_jetson.sh llama
# in another shell
curl http://localhost:8001/healthz

# or run DeepSeek instead
./scripts/run_jetson.sh deepseek
curl http://localhost:8002/healthz
```

## Benchmark both services

Running both models concurrently may exceed the practical memory envelope on an 8 GB Orin Nano. If that happens, benchmark sequentially.

```bash
./scripts/run_jetson.sh benchmark
```

Outputs are saved under `outputs/`.

## API examples

### Generic generation

```bash
curl -X POST http://localhost:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain edge-first inference for marine DX telemetry.",
    "max_new_tokens": 128,
    "temperature": 0.1
  }'
```

### OpenAI-compatible chat route

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role":"system","content":"You are a marine network copilot."},
      {"role":"user","content":"Give a 5-step response plan for packet loss in a polar gateway."}
    ]
  }'
```

## Dataset design

The synthetic dataset intentionally targets these domain concepts:
- marine DX
- edge AI / physical AI
- network transport under intermittent or harsh conditions
- latency-aware operational actions
- later migration to larger DGX-class training or evaluation environments

Generated files:
- `dataset/generated/train.jsonl`
- `dataset/generated/val.jsonl`
- `dataset/generated/test.jsonl`
- `dataset/generated/manifest.json`

## DGX Spark migration path

```bash
./scripts/package_for_dgx.sh
```

Then on the DGX side:

```bash
tar -xzf marine-dx-slm-stack-dgx-portable.tar.gz
cp .env.example .env
# or use release/.env.dgx.example as a starting point
# adjust HF_TOKEN and model settings
```

## Notes on operational tuning

- Start with one model container at a time on Orin Nano.
- Keep `max_new_tokens` small for latency control.
- Preserve `outputs/` and `dataset/generated/` so the same prompts and data can be reused on DGX Spark.
- If you later add LoRA or QLoRA, keep the dataset schema unchanged so fine-tuning, evaluation, and prompt-only baselines remain comparable.
