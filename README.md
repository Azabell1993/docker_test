# Jetson SLM Stack (Marine DX Physical AI)

- Jetson Orin Nano 환경용 Docker Compose 기반 소형 언어모델 추론 서버 스택
- 서비스별 분리 실행 지원: Llama, DeepSeek, Qwen
- Jetson Unified Memory 제약 대응 목적
- CUDA allocator 문제 완화 목적
- 하이브리드 GPU offload 사용
- 네이티브 CUDA 프로브 사용

## 목차

1. [개요](#개요)
2. [플랫폼 특성](#플랫폼-특성)
3. [지원 모델](#지원-모델)
4. [아키텍처](#아키텍처)
5. [네이티브 `.cpp` / `.cu` 역할](#네이티브-cpp--cu-역할)
6. [개선 리포트](#개선-리포트)
7. [사전 준비](#사전-준비)
8. [설정](#설정)
9. [실행](#실행)
10. [API](#api)
11. [테스트](#테스트)
12. [트러블슈팅](#트러블슈팅)
13. [DGX 대응](#dgx-대응)
14. [요약](#요약)
15. [CLI 빠른 참조](#cli-빠른-참조)
16. [벤치마크 결과](#벤치마크-결과-meta-llamalama-32-1b-instruct)

## 개요

### 현재 상태

| 항목 | 상태 |
|------|------|
| 지원 모델 | `llama`, `deepseek`, `qwen` |
| API 서버 | FastAPI + uvicorn |
| 테스트 스크립트 | `test_slm.sh`에서 서버 시작, API 검증, 성능 리포트 수행 |
| GPU 사용 방식 | 전체 GPU 적재 대신 저메모리 `cuda-offload-*` 하이브리드 모드 우선 |
| CPU 폴백 | CUDA 배치 실패 시 `cpu-bfloat16` 폴백 |
| 네이티브 확장 | `safe_ops` (`.cpp` + `.cu`) 빌드 후 서버에서 사용 |
| 데이터셋 준비 | `6G Network Slicing QoS` prep scaffold만 유지 |

### 데이터셋 준비 상태

| 항목 | 내용 |
|------|------|
| 대상 데이터셋 | `6G Network Slicing QoS (네트워크 슬라이싱)` |
| 출처 | `https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset` |
| 원본 형식 | `CSV` |
| 핵심 특징 | `throughput`, `latency`, `packet_loss_rate` |
| 목표 용도 | `QoS prediction`, `slice optimization` |
| LLM 활용 | `지연에 민감한 슬라이스 분석`, `QoS 이상 탐지 질의` |
| 현재 반영 범위 | raw/prepared 디렉터리와 prep manifest만 추가 |

## 플랫폼 특성

| 항목 | 값 |
|------|----|
| 플랫폼 | `NVIDIA Jetson Orin Nano` |
| 베이스 이미지 | `nvcr.io/nvidia/pytorch:24.12-py3-igpu` |
| 아키텍처 | `aarch64 (arm64)` |
| 메모리 구조 | Unified Memory, CPU/GPU 공유 |
| 주요 제약 | GPU free memory 숫자와 실제 `cudaMalloc` 가능 용량이 다를 수 있음 |

Jetson에서는 다음 현상이 중요합니다.

1. `torch.cuda.is_available()`가 `true`여도 실제 모델 배치가 실패할 수 있습니다.
2. `device_map="auto"`만으로는 allocator assert 또는 `NvMapMemAllocInternalTagged` 오류가 날 수 있습니다.
3. 따라서 실제 할당 가능한 GPU budget을 먼저 찾고, 그 budget 안에서만 일부 레이어를 GPU에 올리는 전략이 필요합니다.

## 지원 모델

| 모델 키 | 모델 ID | 포트 | 비고 |
|--------|---------|------|------|
| `llama` | `meta-llama/Llama-3.2-1B-Instruct` | `8000` | 현재 GPU offload 검증의 기준 모델 |
| `deepseek` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `8001` | 1.5B급, 메모리 압박 더 큼 |
| `qwen` | `Qwen/Qwen2.5-1.5B-Instruct` | `8002` | multi-service 테스트용 |

## 아키텍처

- 핵심 경로: 아래와 같음

1. `jetson_slm_stack/app/server.py`
  모델 로드, GPU offload, CPU 폴백, 생성 API, 헬스체크 담당
2. `jetson_slm_stack/app/csrc/safe_ops.cpp`
  Python과 네이티브 코드를 연결하는 pybind 바인딩
3. `jetson_slm_stack/app/csrc/safe_ops_cuda.cu`
  실제 `cudaMalloc` 기반 메모리 프로브와 llama split 계산 담당
4. `jetson_slm_stack/docker/Dockerfile.jetson`
  이미지 빌드 중 `safe_ops`를 컴파일
5. `test_slm.sh`
  파이프 입력, JSON-safe 응답 출력, `/generate`와 `/v1/chat/completions` 성능 리포트 제공

## 네이티브 `.cpp` / `.cu` 역할

- 네이티브 코드는 단순 부가 기능이 아니라 GPU 안정화에 직접 관여

### `.cpp`의 역할

파일: `jetson_slm_stack/app/csrc/safe_ops.cpp`

1. Python에서 호출할 수 있는 모듈 `safe_ops`를 정의합니다.
2. `configure_runtime()`로 PyTorch CPU thread 설정을 적용합니다.
3. `.cu`에서 구현한 `get_cuda_mem_info`, `probe_cuda_budget`, `probe_llama_cuda_split`을 Python으로 노출합니다.

- 요약: `.cpp`는 서버와 네이티브 코드를 연결하는 브리지 역할

### `.cu`의 역할

파일: `jetson_slm_stack/app/csrc/safe_ops_cuda.cu`

1. `cudaMemGetInfo()`로 free/total memory를 읽습니다.
2. `try_alloc_mb()`로 특정 크기 메모리를 실제 할당해봅니다.
3. `probe_cuda_budget()`으로 안전한 GPU 예산을 계산합니다.
4. `probe_llama_cuda_split()`으로 llama에서 GPU에 올릴 수 있는 레이어 수를 보수적으로 산정합니다.

- 요약: `.cu`는 “GPU가 실제로 감당 가능한 만큼만 쓰게 만드는 판단 엔진” 역할

## 개선 리포트

### 개선 배경

- Jetson 환경 특징:
  `torch.cuda.is_available()`가 `true`여도 실제 서비스가 안정적으로 GPU 추론을 수행하는 것은 아님
- 주요 원인:
  Unified Memory 구조
- 주요 원인:
  Jetson allocator 특성
- 반복 발생 문제:

1. free memory 수치상으로는 충분해 보여도 실제 `cudaMalloc` 단계에서 실패
2. `device_map="auto"` 사용 시 allocator assert 또는 `NvMapMemAllocInternalTagged` 오류 발생
3. 모델 로드는 되었지만 첫 `generate()` 시점에서 500 에러 발생
4. 서버의 실제 오류와 테스트 스크립트의 JSON 파싱 오류가 섞여 원인 분리가 어려움

- 대응 방향:
  Python 추정만으로 GPU를 사용하지 않도록 변경
- 대응 방향:
  네이티브 `.cpp` / `.cu` 코드 추가
- 대응 방향:
  실제 CUDA 상태를 직접 확인하도록 변경

### 네이티브 코드 추가 내용

#### 파일: `jetson_slm_stack/app/csrc/safe_ops.cpp`

`.cpp`는 Python 서버와 네이티브 기능 사이의 연결 계층 용도로 작성하였으며, 이 파일이 필요한 이유는 아래와 같다.
- `.cpp` 작성 목적:
  Python 서버와 네이티브 기능 사이의 연결 계층 구성
- 필요 이유:

1. Python에서 CUDA 관련 저수준 기능을 안정적으로 호출할 수 있는 `safe_ops` 모듈을 제공하도록 설계
2. `.cu`에서 구현한 CUDA 메모리 프로브 로직을 서버에서 직접 재사용할 수 있게 구성
3. CPU fallback 시 PyTorch thread 수를 일관되게 제어할 수 있도록 반영

**구현 정의**

1. Python import 가능한 `safe_ops` 모듈 정의
2. `configure_runtime()`로 CPU intra-op / inter-op thread 설정 적용
3. `get_cuda_mem_info`, `probe_cuda_budget`, `probe_llama_cuda_split`를 Python으로 노출

- 해석:
  `.cpp`는 단독 성능 향상 코드라기보다 서버가 네이티브 CUDA 판단 로직을 사용할 수 있게 만드는 bridge 성격

#### 파일: `jetson_slm_stack/app/csrc/safe_ops_cuda.cu`

`.cu`는 실제 GPU 안정화 판단을 수행하는 핵심 코드로 작성하였으며, 이 파일이 필요한 이유는 아래와 같다.
- `.cu` 작성 목적:
  실제 GPU 안정화 판단 수행
- 필요 이유:

1. Jetson에서는 free memory 수치와 실제 할당 가능 메모리가 다를 수 있음
2. 따라서 Python에서 단순히 숫자만 읽는 방식으로는 안전한 GPU budget을 판단하기 어려움
3. 실제 `cudaMalloc`을 시도해봐야 “지금 이 환경에서 몇 MB까지 안전하게 쓸 수 있는지” 알 수 있음

** 구현 정의**

1. `cudaMemGetInfo()`로 free / total memory 조회
2. `try_alloc_mb()`로 특정 용량을 실제 할당해 안전성 확인
3. `probe_cuda_budget()`으로 안전한 GPU budget 계산
4. `probe_llama_cuda_split()`으로 llama 레이어 중 GPU에 올릴 수 있는 범위를 보수적으로 산정

- 해석:
  `.cu`는 “GPU가 실제로 감당 가능한 만큼만 쓰게 만드는 판단 엔진”

### 코드 최적화 내용

1. Jetson FSDP import 문제를 우회해 `generate()` 내부 500 가능성을 제거
2. GPU 전체 적재 대신 `cuda-offload-*` 하이브리드 경로를 우선 사용하도록 변경
3. GPU 예산을 Python 추정치가 아니라 네이티브 CUDA 프로브 결과로 결정
4. 실패 시 서버를 죽이지 않고 `cpu-bfloat16` 폴백으로 생존성 확보
5. `test_slm.sh`가 비JSON 응답에도 깨지지 않도록 변경해 실제 서버 오류와 파싱 오류를 분리

### 각 최적화의 필요성과 개선점

#### 1. Jetson FSDP import 우회
1. Jetson NGC PyTorch 빌드에는 일부 distributed/FSDP 경로가 비어 있음
2. `transformers.generate()` 내부에서 해당 경로가 호출되면 서비스가 500으로 실패할 수 있음

- 반영한 점:
1. 모델은 떠 있는데 생성 API만 죽는 문제를 줄임
2. `/generate`와 `/v1/chat/completions`의 기본 안정성 확보

#### 2. 전체 GPU 적재 대신 하이브리드 offload 우선
1. Jetson 환경에서는 모델 전체를 GPU로 올리는 시도가 allocator 실패를 유발하기 쉬움
2. 일부 레이어만 GPU에 두는 방식이 더 현실적이고 재현성이 높음

- 반영한 점:
1. `active_device=cuda` 상태를 유지하면서도 OOM 확률 감소
2. 전체 실패 대신 부분 GPU 활용 가능
3. `cuda-offload-640mb` 같은 명시적 로드 모드로 현재 상태 확인 가능

#### 3. Python 추정 대신 네이티브 CUDA 프로브 기반 budget 결정
1. free memory 수치만으로는 Jetson allocator의 실제 동작을 설명하기 어려움
2. 서비스 시작 시 보이는 메모리와 실제 로딩 시 쓸 수 있는 메모리가 다를 수 있음

- 반영한 점:
1. “보이기만 하는 메모리”가 아니라 “실제로 할당 가능한 메모리” 기준으로 로드
2. 로드 전략이 추정 기반에서 실측 기반으로 전환
3. GPU 사용 안정성 향상

#### 4. `cpu-bfloat16` 폴백 추가
1. GPU 로드 실패를 서버 전체 실패로 두면 운영 안정성이 크게 떨어짐
2. Jetson 환경에서는 일시적 allocator 실패가 발생할 수 있으므로, 살아있는 fallback 경로가 필요함

- 반영한 점:
1. GPU 실패 시에도 서버 API 유지
2. 단순 다운 대신 응답 가능한 CPU 모드로 전환
3. `load_mode`를 통해 실제 동작 모드 추적 가능

#### 5. `test_slm.sh` 파싱 안정화
1. 이전에는 비JSON 응답을 무조건 `jq`에 넘겨 `parse error`가 먼저 발생함
2. 이 때문에 서버 자체의 500과 스크립트의 파싱 실패가 섞여 보였음

- 반영한 점:
1. HTTP 상태 코드와 응답 본문을 분리 수집
2. JSON일 때만 `jq`로 파싱
3. 비JSON 응답은 그대로 출력
4. 실제 서버 오류와 테스트 도구 오류를 명확히 구분 가능

### 반영 결과
1. Jetson에서 GPU 사용 여부를 추정이 아니라 실측 기준으로 판단
2. 전체 GPU 적재 실패를 줄이고 하이브리드 GPU 사용 경로 확보
3. GPU 실패 시에도 CPU `bfloat16` 폴백으로 서버 생존성 유지
4. 생성 API의 Jetson 환경 500 가능성 감소
5. 테스트 스크립트를 통해 오류 원인 분리가 쉬워짐
6. `healthz`의 `active_device`, `model_device`, `load_mode`를 통해 실제 런타임 상태를 명확히 확인 가능

### 반영하기 전 겪은 이슈
1. Python이 free memory 수치만 보고 GPU 사용 여부를 잘못 판단
2. 로드 시점 또는 첫 generate 시점에 allocator 오류 재발
3. GPU 실패가 곧 서버 다운으로 이어짐
4. 테스트 스크립트에서 `jq parse error`가 먼저 보여 실제 원인 파악이 어려워짐

- 결론:
  단순 성능 튜닝이 아니라 Jetson 환경에서 모델을 실제 서비스 가능한 상태로 만들기 위한 필수 제어 계층으로 반영

---

## 사전 준비

### cuDSS 호스트 설치

- Jetson 호스트에 `cuDSS`가 없으면 컨테이너 내부에서 다음 오류 발생 가능

```text
ImportError: libcudss.so.0: cannot open shared object file
```

- 설치 위치: Jetson 호스트

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudss
```

- 설치 확인:

```bash
find /usr -name 'libcudss.so*' 2>/dev/null
```

## 설정

- 주 설정 파일: `jetson_slm_stack/.env`

### 핵심 변수

| 변수 | 현재 의미 |
|------|-----------|
| `DEVICE` | 요청 디바이스. 일반적으로 `cuda` |
| `DTYPE` | 기본 저장 dtype. 현재 `float16` 기준 경로 사용 |
| `LOAD_IN_4BIT` | Jetson에서는 보통 `0` 유지 |
| `GPU_OFFLOAD_ENABLED` | 저메모리 GPU offload 사용 여부 |
| `GPU_TARGET_MEMORY_MB` | 목표 GPU budget |
| `GPU_MEMORY_RESERVE_MB` | 남겨둘 안전 여유 메모리 |
| `GPU_OFFLOAD_BUFFERS` | offload 버퍼 유지 여부 |
| `GPU_PROBE_STEP_MB` | budget 탐색 감소 단위 |
| `GPU_PROBE_MIN_MB` | 이 값 아래로는 offload 시도하지 않음 |
| `MAX_INPUT_TOKENS` | 최대 입력 토큰 |
| `MAX_NEW_TOKENS` | 최대 생성 토큰 |
| `TEMPERATURE` | 샘플링 강도 |
| `CPU_THREADS` | CPU intra-op thread 수 |
| `CPU_INTEROP_THREADS` | CPU inter-op thread 수 |
| `CPU_FORCE_GREEDY` | CPU 폴백일 때 low-temp 생성에서 greedy 강제 여부 |

### 현재 권장값 예시

- 현재 Llama 기준 안정적으로 동작한 값: 대체로 아래 범위

```env
DEVICE=cuda
DTYPE=float16
LOAD_IN_4BIT=0
GPU_OFFLOAD_ENABLED=1
GPU_TARGET_MEMORY_MB=640
GPU_MEMORY_RESERVE_MB=512
GPU_OFFLOAD_BUFFERS=0
GPU_PROBE_STEP_MB=64
GPU_PROBE_MIN_MB=256
MAX_INPUT_TOKENS=512
MAX_NEW_TOKENS=32
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=0
```

- 적용 방법:
  설정 변경 후 컨테이너 재시작 또는 재빌드 필요

### 설정 해석 포인트

1. `GPU_OFFLOAD_ENABLED=1`은 전체 GPU 적재보다 하이브리드 offload를 우선 사용한다는 뜻입니다.
2. `GPU_TARGET_MEMORY_MB`, `GPU_MEMORY_RESERVE_MB`는 Jetson allocator가 실제로 감당 가능한 범위를 보수적으로 잡기 위한 값입니다.
3. `MAX_INPUT_TOKENS`와 `MAX_NEW_TOKENS`는 입력 길이와 출력 길이를 동시에 제한해 전체 추론 부담을 제어합니다.
4. `TEMPERATURE=0.0`은 사실성 중심의 결정적 출력에 가깝게 동작시키기 위한 설정입니다.

## 실행

### 최초 실행

```bash
./run.sh
```

내부 동작:

1. 모델 다운로드
2. Docker 이미지 빌드
3. 기본 서버 시작

### 재시작

```bash
./rerun.sh
```

### 특정 모델 실행

```bash
./jetson_slm_stack/scripts/run_jetson.sh llama
./jetson_slm_stack/scripts/run_jetson.sh deepseek
./jetson_slm_stack/scripts/run_jetson.sh qwen
```

### 모델별 단축 스크립트

```bash
./start_server.sh llama
./start_server.sh qwen --rebuild
./stop_server.sh llama
./stop_server.sh all
```

## API

- 두 API 공통점:
  같은 모델 호출
- 두 API 차이점:
  입력 형식과 프롬프트 구성 방식이 다름
- 결과 차이:
  응답 구조와 prompt token 수가 달라질 수 있음

### `/healthz`

```bash
curl -s http://localhost:8000/healthz | jq
```

- 확인해야 할 핵심 필드:

1. `requested_device`: 사용자가 요청한 디바이스
2. `active_device`: 실제 런타임 디바이스
3. `model_device`: 모델의 실제 배치 디바이스
4. `load_mode`: `cuda-offload-640mb`, `cpu-bfloat16` 같은 실제 로드 모드
5. `cuda_memory_allocated`, `cuda_memory_reserved`: 실제 CUDA 사용량

### `/generate`

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is edge AI?","max_new_tokens":32,"temperature":0.0,"top_p":1.0}' | jq
```

### `/v1/chat/completions`

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is edge AI?"}],"max_new_tokens":32,"temperature":0.0,"top_p":1.0}' | jq
```

## 테스트

- `test_slm.sh` 수행 항목:

1. 서버가 없으면 대상 서비스만 재시작
2. `/healthz` 확인
3. `/v1/models` 확인
4. `/generate` 호출
5. `/v1/chat/completions` 호출
6. 성능 리포트 출력

### 예시

```bash
./test_slm.sh llama
./test_slm.sh deepseek
./test_slm.sh qwen
printf '%s\n' 'Did King Sejong create Hangul while using a MacBook?' | ./test_slm.sh llama --rebuild
```

### 스크립트 변경점

- 현재 `test_slm.sh` 지원 항목:

1. 파이프 입력과 대화형 입력 둘 다 지원
2. HTTP 상태 코드와 본문을 분리 수집
3. 비JSON 응답도 그대로 출력
4. JSON 응답일 때만 `jq`로 파싱

- 효과:
  스크립트 자체의 `jq parse error`와 서버의 실제 500 오류를 구분 가능

## 트러블슈팅

| 증상 | 원인 | 대응 |
|------|------|------|
| `libcudss.so.0 not found` | 호스트 cuDSS 미설치 | 호스트에 `cudss` 설치 |
| `torch.cuda.is_available() = False` | NVIDIA runtime 설정 누락 | Compose의 `runtime: nvidia`와 환경 변수 확인 |
| `NvMapMemAllocInternalTagged error 12` | Jetson allocator가 큰 contiguous 할당 실패 | GPU budget 축소, offload 사용, reserve 확대 |
| `INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp` | free memory 수치와 실제 할당 가능 메모리 불일치 | 네이티브 probe 결과 기준으로 `GPU_TARGET_MEMORY_MB` 조정 |
| 응답은 오는데 `active_device=cpu` | CUDA 로드 실패 후 폴백 | `load_mode`, 로그, `GPU_*` 변수 확인 |
| `parse error: Invalid numeric literal` | 스크립트가 비JSON 응답을 `jq`에 바로 전달 | 최신 `test_slm.sh` 사용 |
| 4-bit 양자화 실패 | Jetson Orin iGPU에서 bitsandbytes 제약 | `LOAD_IN_4BIT=0` 유지 |

## DGX 대응

```bash
./jetson_slm_stack/scripts/package_for_dgx.sh
```

- 생성물 위치: `jetson_slm_stack/release`
- 주의:
  `models/`는 포함되지 않음
- 결과:
  DGX 환경에서 별도 다운로드 필요

## 요약

이 저장소의 현재 핵심은 다음입니다.

1. Jetson에서 작은 모델을 안정적으로 추론하기 위한 하이브리드 GPU offload
2. Python 추정이 아닌 네이티브 CUDA 메모리 프로브 기반 로드 전략
3. 실패 시 CPU `bfloat16` 폴백으로 서버 생존성 유지
4. `test_slm.sh`를 통한 일관된 재현과 성능 리포트

## CLI 빠른 참조

### 빌드

```bash
docker compose down
docker compose build --no-cache
```

### 의존성 확인

```bash
docker compose run --rm llama32-server \
  python3 -c "import torch; import transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 최초 실행

```bash
./run.sh
```

### 재시작 (코드 수정 후)

```bash
./rerun.sh
```

### 서버 실행 후 상태 확인

```bash
./jetson_slm_stack/scripts/run_jetson.sh llama
curl http://localhost:8000/healthz
```

### 프롬프트 테스트

```bash
echo 'What is edge AI?' | ./test_slm.sh llama
```
