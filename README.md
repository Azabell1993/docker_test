# Jetson SLM Stack (Marine DX Physical AI)

Jetson Orin Nano 환경에서 Docker 기반으로 SLM(LLaMA / DeepSeek)을 구동하기 위한 추론 서버 스택입니다.

---

## 플랫폼 사양

| 항목 | 값 |
|------|----|
| 플랫폼 | NVIDIA Jetson Orin Nano |
| 베이스 이미지 | `nvcr.io/nvidia/pytorch:24.12-py3-igpu` |
| 아키텍처 | aarch64 (arm64) |
| Unified Memory | 8GB (CPU/GPU 공유) |
| GPU 사용 메모리 | ~2,374 MB (모델 로드 시, 31.2%) |
| 추론 속도 | ~17 tok/s (CUDA float16) |

---

## 지원 모델

| 모델 | 파라미터 | 정밀도 | VRAM |
|------|---------|--------|------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | float16 | ~2.4 GB |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | float16 | ~3.0 GB |

> **3B 이상은 로드 시 OOM 위험, 7B 이상은 불가.**  
> Jetson Orin iGPU(sm_87)에서 `bitsandbytes` 미지원으로 4-bit 양자화 불가.

---

## 사전 준비: cuDSS 호스트 설치

Jetson에서 Docker 컨테이너의 CUDA는 호스트 라이브러리에 의존합니다.  
cuDSS가 호스트에 없으면 다음 에러가 발생합니다:

```
ImportError: libcudss.so.0: cannot open shared object file
```

**설치 명령 (Jetson 호스트에서 실행):**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudss
```

**설치 확인:**

```bash
find /usr -name 'libcudss.so*' 2>/dev/null
```

---

## 설정

### .env 파일 (`jetson_slm_stack/.env`)

주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `DEVICE` | `cuda` | 추론 디바이스. `cuda` 또는 `cpu` |
| `DTYPE` | `float16` | 모델 정밀도. Jetson에서는 `float16` 권장 |
| `LOAD_IN_4BIT` | `0` | **반드시 0** — Jetson sm_87에서 bitsandbytes 미지원 |
| `MAX_INPUT_TOKENS` | `1024` | 입력 최대 토큰 수 |
| `MAX_NEW_TOKENS` | `512` | 생성 최대 토큰 수 |
| `TEMPERATURE` | `0.2` | 생성 다양성. 낮을수록 사실 위주 응답 |
| `TOP_P` | `0.9` | Nucleus Sampling 누적 확률 |
| `TOP_K` | `40` | 후보 토큰 수 상한 |
| `REPETITION_PENALTY` | `1.05` | 반복 억제 (1.0 = 억제 없음) |
| `ENABLE_WARMUP` | `1` | 서버 시작 시 GPU 예열 추론 실행 |
| `EMPTY_CACHE_ON_OOM` | `1` | OOM 발생 시 CUDA 캐시 자동 정리 후 재시도 |

> `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8`  
> Jetson Unified Memory 최적화용. `expandable_segments:True`는 Jetson nvmap 미지원으로 제거됨.

### HuggingFace Token

```bash
# jetson_slm_stack/.env
HF_TOKEN=hf_***
```

---

## 실행

### 최초 실행 (모델 다운로드 포함)

```bash
./run.sh
```

내부 동작: 모델 다운로드 → 컨테이너 빌드 → SLM 서버 시작

### 재시작 (서버 코드 수정 후)

```bash
./rerun.sh
```

메모리 정리 → 컨테이너 재시작. 서버는 `server.py`를 메모리에 올려두므로 코드 수정 시 반드시 재시작 필요.

### 특정 모델 실행

```bash
./jetson_slm_stack/scripts/run_jetson.sh llama      # Llama-3.2-1B-Instruct
./jetson_slm_stack/scripts/run_jetson.sh deepseek   # DeepSeek-R1-Distill-Qwen-1.5B
./jetson_slm_stack/scripts/run_jetson.sh llama -d   # 백그라운드 실행
```

---

## API

두 엔드포인트는 내부적으로 동일한 `generate_text()`를 공유하며, 앞단 처리 방식만 다릅니다.

### `/generate` — Raw 문자열 입력

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is edge AI?", "max_new_tokens": 100}'
```

- `prompt` 문자열을 그대로 모델에 전달
- system prompt 없음, chat template 미적용
- 문장 이어쓰기 방식으로 응답

### `/v1/chat/completions` — 대화형 입력

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is edge AI?"}]}'
```

- `messages` 배열을 `build_prompt()`로 변환
- system prompt 자동 삽입 + `tokenizer.apply_chat_template()` 적용
- instruction-tuned 모델에 최적화된 완성도 높은 응답

> **System Prompt 제한 주의:**  
> `SYSTEM_PROMPT`로 도메인을 제한해도 1B 모델은 RLHF 특성상 거절 지시를 일관되게 따르지 않습니다.  
> 도메인 제한이 필요한 경우 7B 이상 모델이 필요하며, 현 플랫폼에서는 지원 불가입니다.

### 헬스체크

```bash
curl http://localhost:8000/healthz
```

---

## 테스트

### 빠른 서버 상태 + 성능 리포트

```bash
./test_slm.sh llama
./test_slm.sh deepseek
```

### 데이터셋 일괄 평가

```bash
./test_dataset.sh    # 출력: test_slm_output/
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `libcudss.so.0 not found` | 호스트에 cuDSS 미설치 | 사전 준비 섹션 참고 |
| `torch.cuda.is_available() = False` | docker runtime 설정 누락 | `docker-compose.yml`에 `runtime: nvidia` 확인 |
| 컨테이너 OOM (exit 137) | Unified Memory 부족 | `./rerun.sh`로 재시작, 필요시 Jetson 리부트 |
| `NvMapMemAllocInternalTagged error 12` | `expandable_segments:True` 설정 | `.env`에서 해당 설정 제거 |
| 4-bit 양자화 에러 | Jetson sm_87 bitsandbytes 미지원 | `LOAD_IN_4BIT=0` 유지 |
| 서버 코드 수정 후 반영 안 됨 | uvicorn `--reload` 없음 | `./rerun.sh`로 컨테이너 재시작 |

---

## 확장 (DGX 대응)

```bash
./jetson_slm_stack/scripts/package_for_dgx.sh
```

현재 구조(Docker + 로컬 모델 캐시 + HuggingFace 의존 제거 가능)를 그대로 DGX 환경으로 이식 가능합니다.

---

## 구조 요약

| 항목 | 값 |
|------|----|
| 추론 서버 | FastAPI + uvicorn (`app/server.py`) |
| 모델 로드 방식 | CPU 로드 → CUDA 이동 (2단계, OOM 안정화) |
| 메모리 관리 | 추론 전후 `gc.collect()` + `torch.cuda.empty_cache()` |
| 설정 관리 | `jetson_slm_stack/.env` 단일 파일 |
| 확장성 | DGX / 클라우드 이식 가능 |


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
  
---  



## 벤치마크 결과 (`meta-llama/Llama-3.2-1B-Instruct`)

**테스트 질문:** `"Did King Sejong create Hangul while using a MacBook?"`  
*(의도적으로 틀린 전제 포함 — 팩트 교정 지능 검증)*

### .env 설정 (테스트 시점)

| 파라미터 | 값 | 용도 |
|----------|----|------|
| `DEVICE` | `cuda` | 추론 디바이스. CPU 대비 ~5배 이상 빠름 |
| `DTYPE` | `float16` | 모델 정밀도. float32 대비 메모리 절반, 속도 향상 |
| `LOAD_IN_4BIT` | `0` | 4-bit 양자화. Jetson sm_87 미지원으로 반드시 0 |
| `MAX_INPUT_TOKENS` | `1024` | 입력 프롬프트 최대 토큰 수. 초과 시 자동 truncation |
| `MAX_NEW_TOKENS` | `512` | 생성할 응답 최대 토큰 수. 길수록 상세하지만 지연 증가 |
| `TEMPERATURE` | `0.2` | 응답 다양성. 낮을수록 사실 위주, 높을수록 창의적 |
| `TOP_P` | `0.9` | 누적 확률 상위 90% 토큰 중 선택 (Nucleus Sampling) |
| `TOP_K` | `40` | 후보 토큰 수 상한. TOP_P와 병행 적용 |
| `REPETITION_PENALTY` | `1.05` | 동일 단어 반복 억제. 1.0 = 억제 없음 |

### API별 결과

| 항목 | `/generate` | `/v1/chat/completions` |
|------|-------------|------------------------|
| Prompt 토큰 | 13 | 62 (+49, system prompt + chat template) |
| Completion 토큰 | 258 | 233 |
| 응답 시간 | 15.542 sec | 15.027 sec |
| **처리 속도** | **16.6 tok/s** | **16.553 tok/s** |
| GPU Memory | 2,374 MB / 7,620 MB (31.2%) | 2,374 MB / 7,620 MB (31.2%) |
| 팩트 오류 교정 | ✅ | ✅ |
| 응답 완성도 | 중 (문장 이어쓰기) | 높음 (instruction 최적화) |

> 상세 결과: [`[GPU]max_input_tokens 1024, max_new_tokens_default 512 King Sejong English Question Test.md`]([GPU]max_input_tokens%201024%2C%20max_new_tokens_default%20512%20King%20Sejong%20English%20Question%20Test.md)
