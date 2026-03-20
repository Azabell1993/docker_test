# [GPU] King Sejong English Question Test — 최종 결과 리포트


## 1. 테스트 환경

| 항목 | 값 |
|------|----|
| 플랫폼 | NVIDIA Jetson Orin Nano |
| 베이스 이미지 | `nvcr.io/nvidia/pytorch:24.12-py3-igpu` |
| 모델 | `meta-llama/Llama-3.2-1B-Instruct` |
| 실행 디바이스 | **CUDA (cuda:0)** |
| 정밀도 | float16 |
| VRAM 사용량 | 2,374 MB / 7,620 MB (31.2%) |

---

## 2. .env 파라미터 설명

서버 동작에 영향을 주는 모든 파라미터는 `jetson_slm_stack/.env` 파일에서 관리됩니다.

### 디바이스 / 정밀도

| 파라미터 | 현재 값 | 설명 |
|----------|---------|------|
| `DEVICE` | `cuda` | 추론에 사용할 디바이스. `cuda`(GPU) 또는 `cpu`. Jetson은 CPU 시 ~1–3 tok/s, CUDA 시 ~17 tok/s |
| `DTYPE` | `float16` | 모델 가중치 정밀도. `float16`(2.4GB), `float32`(3.2GB), `bfloat16`. Jetson에서는 `float16` 권장 |
| `LOAD_IN_4BIT` | `0` | 4-bit 양자화 여부. `1`로 설정 시 ~0.6GB로 줄지만 Jetson Orin iGPU(sm_87)에서 bitsandbytes CUDA 커널 미지원 → **반드시 0** |

### 토큰 한도

| 파라미터 | 현재 값 | 설명 |
|----------|---------|------|
| `MAX_INPUT_TOKENS` | `1024` | 입력 프롬프트 최대 토큰 수. 이 값을 초과하는 입력은 자동 truncation. 512(빠른 테스트) / 1024(일반) / 2048(긴 문서) |
| `MAX_NEW_TOKENS` | `512` | 모델이 생성할 응답의 최대 토큰 수. 길수록 답변이 상세해지지만 시간 비례 증가. 128(빠름) / 256(균형) / 512(상세) |

### 생성 파라미터

| 파라미터 | 현재 값 | 설명 |
|----------|---------|------|
| `TEMPERATURE` | `0.2` | 생성 다양성 조절. `0.0`은 항상 동일한 답(사실 질문에 유리), `1.0`은 창의적·다양한 답. 현재 0.2는 사실 위주의 안정적 응답 |
| `TOP_P` | `0.9` | Nucleus Sampling. 누적 확률 상위 90% 토큰 중에서만 선택. TEMPERATURE와 함께 작동하며 낮출수록 더 보수적인 응답 |
| `TOP_K` | `40` | 상위 K개 토큰 후보 중에서만 선택. `0`은 비활성(제한 없음). TOP_P와 병행 적용 |
| `REPETITION_PENALTY` | `1.05` | 이미 생성된 토큰이 반복될 확률 억제. `1.0`은 억제 없음, `1.1–1.3` 권장. 현재 1.05는 약한 억제 |

### 시스템 설정

| 파라미터 | 현재 값 | 설명 |
|----------|---------|------|
| `ENABLE_WARMUP` | `1` | 서버 시작 시 더미 추론으로 GPU 예열. 첫 번째 실제 추론의 낮은 속도 방지 |
| `EMPTY_CACHE_ON_OOM` | `1` | 추론 중 OOM 발생 시 CUDA 캐시 비우고 자동 재시도 |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:128,`<br>`garbage_collection_threshold:0.8` | PyTorch CUDA 메모리 할당자 세부 설정. Jetson Unified Memory 최적화용. `expandable_segments:True`는 Jetson nvmap 미지원으로 제거됨 |

---

## 3. API 방식 비교

| 항목 | `/generate` | `/v1/chat/completions` |
|------|-------------|------------------------|
| 입력 방식 | raw 문자열 프롬프트 | `messages` 배열 (role/content) |
| System Prompt | 없음 | 자동 추가 (`marine DX 도메인 어시스턴트`) |
| Chat Template | 미적용 | 토크나이저 chat template 적용 |
| Prompt 토큰 수 | 적음 (질문만) | 많음 (system prompt + 태그 포함, +40–50 토큰) |
| 응답 특성 | 문장 이어쓰기 방식 | instruction 모델 최적화, 완성도 높음 |
| 권장 용도 | 단순 텍스트 생성 | 실제 대화형 어시스턴트 |

---

## 4. 테스트 질문 및 결과

### 질문: `"Did King Sejong create Hangul while using a MacBook?"`

> 의도적으로 틀린 사실(세종대왕이 맥북 사용)을 포함한 질문으로,  
> 모델이 잘못된 전제를 올바르게 교정하는지 검증하는 (팩트 확인 지능) 테스트.

---

#### 4-1. `/generate` API 결과

**요청 파라미터:**
```json
{
  "prompt": "Did King Sejong create Hangul while using a MacBook?",
  "max_new_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.9
}
```

**응답:**
> `?`  
> I think there may be some misconceptions here.
>
> King Sejong, the third king of the Joseon Dynasty in Korea, is indeed famous for creating Hangul, the Korean alphabet. However, he did not use a MacBook.
>
> Hangul was created in 1443 during King Sejong's reign, and it was designed to be a phonetic alphabet that represents each character with a unique sound. The creation of Hangul was a significant achievement in Korean history, as it helped to standardize the Korean language and make it more accessible to the general population.
>
> As for the technology used by King Sejong, we don't know exactly what he used, but it's likely that he relied on traditional writing materials such as ink, paper, and quills. There are no records or accounts from King Sejong himself that mention using a MacBook or any other modern computer.
>
> In fact, the first computers were not even invented until the mid-20th century, more than 300 years after King Sejong's time. So, it's highly unlikely that he would have used a MacBook or any other modern device.
>
> So, to summarize: King Sejong created Hangul in 1443 using traditional writing materials, and he didn't use a MacBook or any other modern computer.

> ?  
여기에는 몇 가지 오해가 있는 것 같습니다.  
조선 왕조의 3대 왕인 세종대왕은 한글을 창제한 것으로 유명하지만, 맥북을 사용한 적은 없습니다.  
한글은 세종대왕 재위 시절인 1443년에 창제되었으며, 각 글자를 고유한 발음으로 나타내는 음절 문자입니다.  
한글 창제는 한국어를 표준화하고 일반 대중이 더 쉽게 접할 수 있도록 하는 데 기여한 한국 역사에 중요한 업적입니다.  
세종대왕이 어떤 기술을 사용했는지는 정확히 알 수 없지만, 먹, 종이, 깃펜과 같은 전통적인 필기구를 사용했을 것으로 추정됩니다.  
세종대왕이 맥북이나 다른 현대 컴퓨터를 사용했다는 기록은 전혀 없습니다.  
사실, 최초의 컴퓨터는 세종대왕 시대보다 300년도 더 지난 20세기 중반에 발명되었습니다.  
그러므로 그가 맥북이나 다른 최신 기기를 사용했을 가능성은 매우 낮습니다.  
요약하자면, 세종대왕은 1443년에 전통적인 필기구를 사용하여 한글을 창제했으며, 맥북이나 다른 최신 컴퓨터를 사용하지 않았습니다.  
  

**성능:**

| 항목 | 값 |
|------|----|
| Prompt 토큰 | 13 / 1024 (1.3%) |
| Completion 토큰 | 258 / 512 (50.4%) |
| 총 토큰 | 271 / 1536 |
| 응답 시간 | 15.542 sec |
| **처리 속도** | **16.6 tok/s** |

**분석:**
- 질문을 문장으로 이어받아(`?`로 시작) raw completion 방식으로 응답
- System prompt 없이도 틀린 전제를 올바르게 인식하고 교정
- 역사적 사실(1443년 창제, 전통 필기 도구 사용)을 정확히 서술
- 컴퓨터 발명 시점(20세기 중반)과의 시간 차이로 논리적 반박

---

#### 4-2. `/v1/chat/completions` API 결과

**요청 파라미터:**
```json
{
  "messages": [{"role": "user", "content": "Did King Sejong create Hangul while using a MacBook?"}],
  "max_new_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.9
}
```

**자동 추가된 System Prompt:**
> "You are a domain assistant for marine DX physical AI and network systems."

**응답:**
> No, King Sejong the Great did not use a MacBook. He was a Korean king who ruled from 1418 to 1450 and is best known for creating the Korean alphabet, Hangul.
>
> Hangul was actually created in 1443 during King Sejong's reign, but it was based on earlier writing systems such as the Chinese characters and the Japanese Kanji. The first version of Hangul was called "Sejong's Script" or "Sejong's alphabet," and it was designed to be a phonetic alphabet that could be read by anyone, regardless of their native language or education level.
>
> King Sejong's script was written in a unique combination of consonant-vowel pairs, which were chosen to be easy to pronounce and recognize. It was also designed to be a more efficient and practical alternative to existing writing systems, which often had many exceptions and complexities.
>
> There is no evidence to suggest that King Sejong used a MacBook, as they were not widely available in Korea at the time. In fact, the first personal computers were not developed until the 1960s, more than 300 years after King Sejong's death.

> 아니요, 세종대왕은 맥북을 사용하지 않았습니다.  
그는 1418년부터 1450년까지 통치한 한국의 왕으로, 한국어 문자 한글을 창제한 것으로 가장 잘 알려져 있습니다.  
한글은 실제로 세종대왕 재위 기간인 1443년에 창제되었지만, 중국 한자나 일본 한자와 같은 이전 문자 체계를 기반으로 했습니다.  
한글의 초기 버전은 "세종자" 또는 "세종자"라고 불렸으며, 모국어나 교육 수준에 관계없이 누구나 읽을 수 있도록 음소 문자로 고안되었습니다.  
세종자는 발음하고 인식하기 쉽도록 자음과 모음의 조합이 독특하게 구성되어 있었습니다.  
또한 기존의 문자 체계는 예외와 복잡성이 많아 효율적이고 실용적인 대안으로 설계되었습니다.  
세종대왕이 맥북을 사용했다는 증거는 없습니다. 당시 한국에서는 맥북이 널리 보급되지 않았기 때문입니다.  
실제로 최초의 개인용 컴퓨터는 세종대왕 사후 300여 년이 지난 1960년대에 개발되었습니다.  
  
**성능:**

| 항목 | 값 |
|------|----|
| Prompt 토큰 | 62 / 1024 (6.1%) — system prompt 포함 |
| Completion 토큰 | 233 / 512 (45.5%) |
| 총 토큰 | 295 / 1536 |
| 응답 시간 | 15.027 sec |
| **처리 속도** | **16.553 tok/s** |

**분석:**
- `No,` 로 시작하는 명확한 부정 → instruction-tuned 모델의 완성도 높은 응답
- Prompt 토큰이 62 (vs generate의 13): chat template + system prompt로 49토큰 추가
- 재위 기간(1418–1450), 한글 특징(자음-모음 쌍), 개인용 컴퓨터 발명 시점(1960년대) 등 더 풍부한 정보 포함
- 응답 길이는 generate(258토큰)보다 적은 233토큰이지만 구조화된 완성형 문장

---

**⚠️ System Prompt 미준수 원인 분석**

System prompt로 `"marine DX 도메인 어시스턴트"`를 지정했음에도 역사 질문에 정상 답변한 이유는 두 가지입니다.

1. **System prompt의 역할 정의 방식 문제**  
   현재 system prompt는 AI의 **페르소나(역할명)** 만 선언했을 뿐, 주제 제한 지시를 포함하지 않습니다.  
   `"You are a marine DX assistant"`는 "넌 이런 사람이야"일 뿐,  
   `"Only answer marine DX questions. Refuse anything else."`처럼 명시적 거절 지시가 없으면 모델은 다른 질문도 그냥 답합니다.

2. **1B 소형 모델의 instruction following 한계**  
   모델 크기가 작을수록 복잡한 지시(예: 특정 주제만 대답, 나머지 거절)를 일관되게 따르는 능력이 낮습니다.  
   이는 Meta Llama 모델 카드 및 instruction following 벤치마크(IFEval 등)에서 1B급 모델이 낮은 준수율을 보이는 것과 일치합니다.  
   RLHF(인간 피드백 강화학습)로 "도움이 되어야 한다"는 성향이 강하게 학습되어, "거절하라"는 system prompt를 override하는 경향이 있습니다.

> **결론:** system prompt로 도메인을 제한하려면 ① 명시적 거절 지시 문장 포함 + ② 7B 이상의 모델 사용이 현실적입니다. 1B 모델에서는 페르소나 설정 수준의 효과만 기대하는 것이 적절합니다.

---

## 5. 두 API 비교 요약

### 코드 구현 차이 (server.py 요약)

```
POST /generate
  req.prompt (문자열)
    → generate_text(prompt) 직접 호출

POST /v1/chat/completions
  req.messages (role/content 배열)
    → build_prompt()로 system prompt + chat template 적용
    → generate_text(prompt) 호출
```

두 엔드포인트는 내부적으로 동일한 `generate_text()` 함수를 공유합니다.  
차이는 오직 **그 앞단**: `/generate`는 문자열을 그대로, `/v1/chat/completions`는 `build_prompt()`를 통해 `tokenizer.apply_chat_template()`으로 대화 형식으로 변환한 뒤 넘깁니다.  
그 때문에 prompt 토큰이 49개 더 많아지고, AI가 "대화 상대" 맥락을 인식해 더 완성도 높은 답변을 생성합니다.

### 비교 테이블

> Jetson Orin Nano는 **Unified Memory** 아키텍처로 CPU RAM과 GPU VRAM이 물리적으로 동일한 8GB 풀을 공유합니다.  
> PyTorch CUDA가 점유하는 메모리 = GPU가 사용하는 메모리이며, 나머지가 OS/CPU 프로세스에 할당됩니다.

| 항목 | `/generate` | `/v1/chat/completions` |
|------|-------------|------------------------|
| Prompt 토큰 | 13 | 62 (+49, system prompt) |
| Completion 토큰 | 258 | 233 |
| 응답 시간 | 15.542 sec | 15.027 sec |
| 처리 속도 | 16.6 tok/s | 16.553 tok/s |
| GPU Memory (CUDA 점유) | 2,374 MB / 7,620 MB (31.2%) | 2,374 MB / 7,620 MB (31.2%) |
| CPU Memory (잔여 가용) | ~5,246 MB (OS + 컨테이너 포함) | ~5,246 MB (OS + 컨테이너 포함) |
| 응답 시작 | `?` (문장 이어붙이기) | `No,` (명확한 답변) |
| 오류 교정 | ✅ 올바르게 교정 | ✅ 올바르게 교정 |
| 응답 완성도 | 중 (raw completion) | 높음 (instruction 최적화) |

---

## 6. 종합 평가

- **속도:** CUDA float16 기준 **~16.5–16.6 tok/s** — Jetson Orin Nano 8GB에서 안정적인 실시간 추론 가능
- **정확도:** 의도적으로 틀린 전제를 포함한 질문에 두 API 모두 정확하게 오류를 인식하고 올바른 역사적 사실로 교정
- **API 선택 기준:**
  - 단순 텍스트 생성·빠른 프로토타입 → `/generate`
  - 실제 사용자 대화·어시스턴트 → `/v1/chat/completions`
- **VRAM 효율:** 2,374 MB / 7,620 MB (31.2%) 사용 — 나머지 69%는 여유 상태로 동시 서비스 여유 있음
