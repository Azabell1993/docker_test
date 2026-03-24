# Marine DX Physical AI + Network SLM Stack for Jetson Orin Nano

Jetson Orin Nano 엣지 디바이스에서 Docker Compose 기반으로 SLM(Small Language Model) 추론 서버를 구동하는 스택입니다.

대상 데이터셋:

- 이름: `6G Network Slicing QoS` (`네트워크 슬라이싱`)
- 형식: `CSV`
- 주요 정형 컬럼 후보: `throughput`, `latency`, `packet_loss_rate`
- 용도: `QoS prediction`, `slice optimization`
- LLM 활용 예시: `지연에 민감한 슬라이스 분석`, `QoS 이상 탐지 질의`
- 출처: `https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset`


## 검증된 실행 환경

| 항목 | 사양 |
|------|------|
| 디바이스 | Jetson Orin Nano |
| 아키텍처 | aarch64 |
| Unified Memory | 8 GB (GPU/CPU 공유) |
| Base Image | `nvcr.io/nvidia/pytorch:24.12-py3-igpu` |
| Python | 3.12 |
| PyTorch | 2.6.0a0 (Jetson NGC build) |
| CUDA | float16, ~2.4 GB VRAM 사용 (전체 7.6 GB 중 31%) |
| 처리량 | ~17 tok/s (`Llama-3.2-1B`, float16) |

## 디렉토리 구조

```
jetson_slm_stack/
├── .env                  ← 실행 파라미터 설정 (직접 수정)
├── .env.example          ← .env 템플릿 (git 추적)
├── docker-compose.yml    ← 서비스 정의
├── app/
│   ├── server.py         ← FastAPI 추론 서버
│   └── download_models.py
├── dataset/
│   ├── raw/network_slicing_qos/
│   ├── prepared/network_slicing_qos/
│   └── scripts/prepare_network_slicing_dataset.py
├── docker/
│   ├── Dockerfile.jetson
│   └── Dockerfile.dgx
├── models/               ← 다운로드된 모델 가중치 (git 제외)
├── outputs/
└── scripts/
    ├── run_jetson.sh
    └── package_for_dgx.sh
```

## 빠른 시작

### 1. 환경 설정

```bash
# .env 파일 확인 및 수정
vi jetson_slm_stack/.env
```

주요 설정값 (`.env`에서 직접 수정):

```env
MAX_INPUT_TOKENS=1024   # 입력 프롬프트 최대 토큰 수
MAX_NEW_TOKENS=512      # 응답 최대 토큰 수 (길수록 느림)
TEMPERATURE=0.2         # 0.0(결정적) ~ 1.0(창의적)
TOP_P=0.9
TOP_K=40
REPETITION_PENALTY=1.05
LOAD_IN_4BIT=0          # 1로 설정 시 VRAM 절약 (정확도 소폭 하락)
```

> **⚠️ 중요**: `MAX_INPUT_TOKENS`, `MAX_NEW_TOKENS`는 `.env`에 반드시 존재해야 합니다.  
> 없을 경우 서버 시작 시 오류 메시지와 함께 즉시 종료됩니다.

### 2. 모델 다운로드

```bash
cd jetson_slm_stack
docker compose run --rm model-download
```

### 3. 서버 실행

```bash
# Llama (포트 8000)
./scripts/run_jetson.sh llama

# DeepSeek (포트 8001)
./scripts/run_jetson.sh deepseek
```

## 테스트 스크립트 (`test_slm.sh`)

루트 디렉토리의 `test_slm.sh`는 서버 시작 + API 테스트 + 성능 리포트를 한번에 수행합니다.

### 사용법

```bash
# 기본 실행 (서버가 올라와 있으면 테스트만)
./test_slm.sh llama
./test_slm.sh deepseek

# 강제 이미지 재빌드 후 재시작
./test_slm.sh llama --rebuild

# 프롬프트를 파이프로 전달 (대화형 입력 건너뜀)
echo 'What is the capital of France?' | ./test_slm.sh llama
echo 'Explain underwater acoustic communication simply.' | ./test_slm.sh deepseek
```

### 동작 방식

| 조건 | 동작 |
|------|------|
| 서버 이미 실행 중 + `--rebuild` 없음 | 테스트만 실행 |
| 서버 없음 또는 `--rebuild` | 해당 서비스만 재시작 (다른 서비스 유지) |
| `--rebuild` | `--no-cache` 빌드 |

### 출력 예시 (성능 리포트)

```
════════════════════════════════════════════════
             PERFORMANCE REPORT
════════════════════════════════════════════════

[ Model ]
  Model ID:                      meta-llama/Llama-3.2-1B-Instruct
  Active Device:                 cuda
  dtype:                         torch.float16

[ Token Limits ]
  Max Input Tokens:              1024 tokens
  Max New Tokens (default):      512 tokens
  Max Total (in+new):            1536 tokens

[ CUDA Memory  (used / total) ]
  Allocated:          2365.3 MB / 7619.9 MB  (31.0%)
  Reserved:           2382.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:      16 / 1024  (1.6%)
  Completion Tokens:  133 / 512  (26.0%)
  Total Tokens:       149 / 1536  (9.7%)
  Latency:            7.496 sec
  Throughput:         17.742 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:      65 / 1024  (6.3%)
  Completion Tokens:  278 / 512  (54.3%)
  Total Tokens:       343 / 1536  (22.3%)
  Latency:            16.589 sec
  Throughput:         17.806 tok/s

════════════════════════════════════════════════
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 서비스 상태 |
| `/healthz` | GET | 모델 상태 + CUDA 메모리 정보 |
| `/v1/models` | GET | 로드된 모델 목록 |
| `/generate` | POST | Raw 텍스트 생성 |
| `/v1/chat/completions` | POST | OpenAI 호환 채팅 API |

### `/generate` vs `/v1/chat/completions` 차이

| 항목 | `/generate` | `/v1/chat/completions` |
|------|-------------|------------------------|
| 입력 | raw 문자열 | `messages` 배열 (role/content) |
| System Prompt | 없음 | 자동 추가 |
| Chat Template | 미적용 | 토크나이저 chat template 적용 |
| Prompt Tokens | 적음 (질문만) | 많음 (system + 태그 포함) |
| 응답 품질 | 단순 completion | instruction 모델 최적화 응답 |
| 권장 용도 | 단순 텍스트 생성 | 실제 챗봇 / 어시스턴트 |

### API 예시

```bash
# Health Check
curl http://localhost:8000/healthz | jq

# Raw 생성
curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain edge-first inference for marine DX telemetry.",
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9
  }' | jq

# OpenAI 호환 채팅
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the main challenges of sonar signal processing in shallow water?"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.2
  }' | jq
```

## 파라미터 튜닝 가이드

`.env` 파일만 수정하고 컨테이너를 재시작하면 적용됩니다.

```bash
# 컨테이너 재시작 (llama 예시)
docker compose -f jetson_slm_stack/docker-compose.yml rm -sf llama32-server
./jetson_slm_stack/scripts/run_jetson.sh llama
```

| 목적 | 권장 설정 |
|------|----------|
| 빠른 테스트 | `MAX_NEW_TOKENS=128`, `MAX_INPUT_TOKENS=512` |
| 균형 | `MAX_NEW_TOKENS=256`, `MAX_INPUT_TOKENS=1024` |
| 코드/긴 답변 | `MAX_NEW_TOKENS=512`, `MAX_INPUT_TOKENS=1024` |
| 사실 질문 | `TEMPERATURE=0.0` |
| 창의적 응답 | `TEMPERATURE=0.7`, `TOP_P=0.95` |
| VRAM 절약 | `LOAD_IN_4BIT=1` |

## 데이터셋

데이터셋은 `6G Network Slicing QoS` 으로 준비함

```
dataset/raw/network_slicing_qos/
├── README.md
└── *.csv

dataset/prepared/network_slicing_qos/
└── manifest.prep.json
```

데이터셋 스크립트: `dataset/scripts/prepare_network_slicing_dataset.py`

## DGX Spark 마이그레이션

```bash
./scripts/package_for_dgx.sh
```

DGX 측

```bash
tar -xzf marine-dx-slm-stack-dgx-portable.tar.gz
cp .env.example .env
# HF_TOKEN, BASE_IMAGE 수정
```

## 개발시 발생하였던 이슈에 대하여 해결한 점

| 문제 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: torch._C._distributed_c10d` | Jetson NGC PyTorch에 distributed 모듈 미포함 | `server.py` 내 FSDP monkey-patch 적용 완료 |
| CUDA OOM (NvMapMemHandleAlloc error 12) | `device_map="auto"` 사용 시 Tegra iGPU 커널 레벨 OOM | CPU 먼저 로드 후 `.to("cuda")` 2단계 방식으로 해결 |
| JSON decode error (프롬프트에 큰따옴표 포함) | shell 변수 직접 삽입 시 JSON 파괴 | `jq -n --arg`로 안전한 JSON 생성 |
| Postman Cloud Agent에서 접근 불가 | 클라우드 서버에서 사설 IP 차단 | Postman Desktop Agent로 전환 |


## Directory layout

- `docker-compose.yml` : compose entrypoint
- `docker/Dockerfile.jetson` : Jetson-target image
- `docker/Dockerfile.dgx` : x86 CUDA / DGX-target image
- `app/server.py` : FastAPI inference service
- `app/download_models.py` : Hugging Face model downloader
- `dataset/scripts/prepare_network_slicing_dataset.py` : 6G Network Slicing QoS prep scaffold
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
