# Jetson SLM Stack (Marine DX Physical AI)
### Jetson Orin Nano 환경에서 Docker 기반으로 SLM (LLaMA / DeepSeek)을 구동하기 위한 환경

---

## 1. cuDSS란?
### cuDSS (CUDA Sparse Solver)
cuDSS는 NVIDIA에서 제공하는 CUDA 기반 **Sparse Linear Solver 라이브러리**입니다.

#### 역할
- Sparse matrix 연산 가속
- 선형 시스템 해결 (Ax = b)
- 물리 시뮬레이션 / 네트워크 / 신호처리 가속

#### 왜 필요한가?
Jetson + PyTorch (특히 최신 CUDA stack)에서는 일부 연산이 내부적으로 cuDSS를 참조합니다.

> **없으면 아래 에러 발생**
> ```
> ImportError: libcudss.so.0: cannot open shared object file
> ```

> **cuDSS는 "옵션"이 아니라 필수 런타임 의존성**

---

## 2. cuDSS 설치

### 공식 다운로드
> https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

### 선택 옵션

| 항목 | 값 |
|------|-----|
| Operating System | LINUX |
| Architecture | aarch64-jetson |
| Compilation | Native |
| Distribution | Ubuntu |
| Version | 22.04 |
| Installer Type | deb (network) |

### 설치 명령

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudss
```

### 설치 확인

```bash
find /usr -name 'libcudss.so*' 2>/dev/null
```

정상 결과:

```
/usr/lib/aarch64-linux-gnu/libcudss.so.0
/usr/lib/aarch64-linux-gnu/libcudss/12/libcudss.so.0.7.1
/usr/lib/aarch64-linux-gnu/libcudss/12/libcudss.so.0
/usr/lib/aarch64-linux-gnu/libcudss/12/libcudss.so
/usr/lib/aarch64-linux-gnu/libcudss.so
```

### 중요 사항: 호스트 설치 필수

> **Jetson 아키텍처 구조**
 - Docker Container CUDA ≠ 완전 독립
 - Host CUDA Library에 의존

> **따라서 cuDSS는 반드시 Jetson OS(호스트)에 설치해야 함**

---

## 3. Docker 설정
### .env 파일 설정

```bash
BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3-igpu
```

> 반드시 이 값이어야 함

### HuggingFace Token

```bash
HF_TOKEN=hf_***
```

https://huggingface.co/settings/tokens 에서 발급

---

## 4. Docker 구성 핵심

### GPU 사용 필수 설정

**docker-compose.yml:**

```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Base Image

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu
```

**이유:**
- Jetson CUDA stack과 완전 호환
- PyTorch + CUDA 사전 구성

---

## 5. 실행

### 전체 실행

```bash
./run.sh
```

**내부 동작 순서:**
1. Dataset 생성
2. 모델 다운로드
3. SLM 서버 실행

### 특정 모델 실행

```bash
./scripts/run_jetson.sh llama
./scripts/run_jetson.sh deepseek
```

---

## 6. 재빌드

```bash
./rebuild.sh
```

또는 수동:

```bash
docker compose down
docker compose build --no-cache
```

---

## 7. 테스트

### 스크립트를 통한 테스트

```bash
./test_slm.sh
```

### Generate API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain marine communication networks",
    "max_new_tokens": 100
  }'
```

### Chat API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is underwater communication?"}
    ]
  }'
```

---

## 8. GPU 동작 확인

```bash
docker compose run --rm llama32-server \
  python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

**정상 출력:**

```
True
1
```

---

## 9. 트러블슈팅

### ❌ libcudss.so.0 에러

**개발시 발생하였던 문제 :**
> ImportError: libcudss.so.0 not found
- 원인: Jetson host에 *cuDSS* 미설치
- 해결: 위의 *cuDSS* 설치 섹션 참고

### ❌ CUDA False
- 문제: `torch.cuda.is_available() = False`
- 원인: docker runtime 설정 없음
- 해결: docker-compose.yml에 `runtime: nvidia` 추가

### ❌ 잘못된 Base Image
- 문제: `l4t-jetpack:r36.4.0` 사용됨
- 해결: `.env` 파일의 `BASE_IMAGE` 수정

---

## 10. 확장 (DGX Spark 대응)

현재 구조는 다음을 만족하는 최초의 형태임
- Docker 기반
- 모델 local cache 구조
- HuggingFace 의존 제거 가능
- GPU runtime 분리

**DGX Spark로 그대로 이식 가능**

---

## 요약

| 항목 | 상태 |
|------|------|
| cuDSS | Host 설치 필수 |
| Docker | PyTorch iGPU image |
| GPU | runtime: nvidia |
| 모델 | local + HF 가능 |
| 확장성 | DGX Spark 대응 |

---

## 11. 메모리 최적화 (Jetson Orin Nano 대응)

### 문제: CUDA Out of Memory 에러

Jetson Orin Nano 같은 저사양 GPU 환경에서 다음 오류 발생 시:

```
RuntimeError: CUDA error: out of memory
NvMapMemHandleAlloc: error 0
```

### 해결 방법

**다음 최적화가 자동 적용됨**

#### 1. 모델 로드 최적화
- `device_map="auto"` 사용으로 CPU 오프로딩 활성화
- 메모리 부족 시 자동 재시도 (CPU로 폴백)
- 선형 체크포인팅(gradient checkpointing) 자동 활성화

#### 2. 생성 최적화
- 기본 토큰 제한 감소: `MAX_NEW_TOKENS: 128` (기존 192)
- 기본 입력 토큰 제한: `MAX_INPUT_TOKENS: 512` (기존 1024)
- 런타임 메모리 오류 시 자동으로 토큰 감소 후 재시도

#### 3. 메모리 정리
- 모델 로드 전후 `gc.collect()` 및 `torch.cuda.empty_cache()` 자동 실행
- 생성 전후 메모리 정리

#### 4. Warmup 개선
- Warmup 토큰 자동 최소화 (4 tokens)
- Warmup 실패 시에도 서버 계속 시작

### 커스터마이징

필요 시 environment 변수로 조정:

```yaml
environment:
  MAX_INPUT_TOKENS: "256"      # 더 작게 줄이기
  MAX_NEW_TOKENS: "64"         # 더 작게 줄이기
  LOAD_IN_4BIT: "0"            # 4비트 양자화 비활성화 유지
  EMPTY_CACHE_ON_OOM: "1"      # OOM 시 캐시 비우기 (권장)
```

---
## CLI TEST
### BUILD
> docker compose down  
> docker compose build --no-cache  

### import 확인
> docker compose run --rm llama32-server \
> python3 -c "import torch; import transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

### 서버 실행
> ./scripts/run_jetson.sh llama
### 서버 뜨면 테스트
> curl http://localhost:8000/healthz
### 서버 실행
> ./scripts/run_jetson.sh llama
### 서버 뜨면 테스트
> curl http://localhost:8000/healthz


네, 요청하신 내용을 바탕으로 **시스템 정보**뿐만 아니라 실제 **API의 Request와 Response 구조**를 명확히 파악할 수 있도록 문서를 다시 정리해 드립니다. 

개발 문서나 작업 리포트로 활용하기 좋은 구성입니다.

---

## LLM API 서버 연동 및 상태 보고서

### 1. 시스템 정보 (Health Check)
서버의 가동 상태와 모델 로드 사양입니다.

* **Endpoint:** `/healthz`
* **Status:** `200 OK` (Server is up!)
* **주요 사양:**
    * **Model:** `Llama-3.2-1B-Instruct` (FP16)
    * **Device:** `NVIDIA GPU (CUDA:0)`
    * **VRAM Usage:** Allocated 2.31GB / Reserved 2.33GB
    * **Config:** Max Input 512 / Max New Tokens 128

---

### 2. API Request & Response 상세
1) DeepSeek

2) llama
```
Server is up!
--- [TEST] Health Check ---
llama32-server-1  | INFO:     172.18.0.1:46636 - "GET /healthz HTTP/1.1" 200 OK
{
  "ok": true,
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "model_source": "./models/meta-llama__Llama-3.2-1B-Instruct",
  "requested_device": "cuda",
  "active_device": "cuda",
  "model_device": "cuda:0",
  "dtype": "torch.float16",
  "cuda_available": true,
  "load_mode": "float16",
  "load_in_4bit_requested": false,
  "load_in_4bit_active": false,
  "bnb_config_available": true,
  "cuda_memory_allocated": 2480148992,
  "cuda_memory_reserved": 2497708032,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
}


--- [TEST] Models List ---
llama32-server-1  | INFO:     172.18.0.1:46648 - "GET /v1/models HTTP/1.1" 200 OK
{
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "owned_by": "local"
    }
  ]
}


--- [TEST] Generation API ---
llama32-server-1  | INFO:     172.18.0.1:36052 - "POST /generate HTTP/1.1" 200 OK
llama32-server-1  | INFO:     172.18.0.1:46652 - "POST /generate HTTP/1.1" 200 OK
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "Paris.\nThe capital of France is Paris. It is located in the north-central part of the country and is known for its rich history, art museums, fashion, and cuisine.\n\nHere are some interesting facts about Paris:\n\n* Paris is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral",
  "prompt_tokens": 8,
  "completion_tokens": 64,
  "total_tokens": 72,
  "latency_sec": 5.502,
  "tokens_per_sec": 11.632
}


--- [TEST] Chat Completions API ---
llama32-server-1  | INFO:     172.18.0.1:36056 - "POST /v1/chat/completions HTTP/1.1" 200 OK
{
  "id": "chatcmpl-local-1773988181780",
  "object": "chat.completion",
  "created": 1773988181,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 57,
    "completion_tokens": 8,
    "total_tokens": 65
  },
  "latency_sec": 1.526,
  "tokens_per_sec": 14.789,
  "device": "cuda",
  "load_mode": "float16"
}
```

### 3. 성능 요약 (Performance Summary)