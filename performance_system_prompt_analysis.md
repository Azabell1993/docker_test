# Performance And System Prompt Analysis

## 범위

이 문서는 다음 두 테스트 결과를 간단한 표로 요약하고, 응답 품질을 평가하며, `/v1/chat/completions` 경로에서 system prompt가 반영되지 않은 것처럼 보인 원인을 추적한 분석 문서이다.

질문 : 
```text
Did King Sejong create Hangul while using a MacBook?
```

비교 대상:

1. `max_new_tokens=32`
2. `max_new_tokens=512`

모델과 런타임 조건:

1. Model: `meta-llama/Llama-3.2-1B-Instruct`
2. Device: `cuda`
3. Load mode: `cuda-offload-640mb`
4. dtype: `torch.float16`

## 1. 성능 테스트 요약

### 1-1. 수치 비교 표

| Case | API | Prompt Tokens | Completion Tokens | Total Tokens | Latency (sec) | Throughput (tok/s) | 핵심 특징 |
|------|-----|---------------|-------------------|--------------|---------------|--------------------|-----------|
| 32 tokens | `/generate` | 13 | 32 | 45 | 20.338 | 1.573 | 짧고 빠른 smoke test, 응답 중간에서 종료 |
| 32 tokens | `/v1/chat/completions` | 62 | 32 | 94 | 19.593 | 1.638 | 더 많은 prompt 문맥, assistant 스타일 응답 |
| 512 tokens | `/generate` | 13 | 260 | 273 | 157.340 | 1.652 | 길어진 설명, 일부 장황함과 환각 포함 |
| 512 tokens | `/v1/chat/completions` | 62 | 189 | 251 | 114.964 | 1.644 | 더 정돈된 형식, 그러나 사실 오류 포함 |

### 1-2. 성능 관찰

1. `/v1/chat/completions`는 항상 prompt token이 더 크다.
2. 이유는 user 질문만 보내는 `/generate`와 달리, chat 경로는 system message와 chat template가 함께 들어가기 때문이다.
3. `max_new_tokens=32`에서는 두 API 모두 짧은 응답 검증에 적합하고 latency 차이가 크지 않았다.
4. `max_new_tokens=512`에서는 실제 생성 토큰 수가 512를 모두 채우지 않았다.
5. 이번 샘플에서는 `/generate`가 260 tokens, `/v1/chat/completions`가 189 tokens에서 멈췄다.
6. 처리량은 두 설정 모두 대략 `1.57 ~ 1.65 tok/s` 범위로 큰 차이가 없었다.
7. 즉 이 환경에서는 `max_new_tokens`를 크게 올려도 순간 처리량이 크게 좋아지지는 않았고, 총 지연시간만 커졌다.

## 2. 지능평가

아래 평가는 단순 성능이 아니라 응답 품질 관점에서의 정성 평가이다.

평가 기준:

1. 사실성: 역사적 사실과 기술 설명이 맞는가
2. 일관성: 답변 구조가 자연스럽고 앞뒤가 맞는가
3. 질문 적합성: 사용자의 질문을 직접적으로 다뤘는가
4. 지시 준수성: system prompt나 대화형 포맷의 의도를 따르는가
5. 환각 위험: 근거 없는 내용을 추가하는가

### 2-1. 품질 평가 표

| Case | API | 사실성 | 구조성 | 질문 적합성 | system prompt 반영감 | 총평 |
|------|-----|--------|--------|-------------|----------------------|------|
| 32 tokens | `/generate` | 보통 이상 | 보통 | 높음 | 낮음 | 질문의 오류를 짧게 바로잡지만 길이 제한으로 불완전 |
| 32 tokens | `/v1/chat/completions` | 보통 이상 | 좋음 | 높음 | 낮음 | 더 자연스러운 assistant 톤, 짧은 정답형 응답 |
| 512 tokens | `/generate` | 보통 | 보통 | 높음 | 낮음 | 핵심 반박은 맞지만 후반에 불필요한 추정과 환각이 섞임 |
| 512 tokens | `/v1/chat/completions` | 낮음 | 좋음 | 높음 | 낮음 | 형식은 가장 자연스럽지만 역사적 사실 오류가 더 두드러짐 |

### 2-2. 상세 평가

#### `32 tokens` 결과

1. 두 API 모두 핵심 의도는 맞췄다.
2. “King Sejong did not use a MacBook”라는 핵심 반박은 적절하다.
3. 그러나 32 tokens 제한 때문에 충분한 근거 제시는 못 하고 중간에서 잘린다.
4. 이 설정은 품질 평가용이라기보다 생존성 확인과 응답 형식 확인에 더 적합하다.

#### `512 tokens` 결과

1. 길어진 응답은 겉보기에는 더 풍부하다.
2. 그러나 길이가 늘면서 사실 오류와 추정성 문장이 증가했다.
3. `/generate`는 “Joseon 시대의 초기 컴퓨터” 같은 비근거성 설명을 덧붙이며 환각 경향이 보인다.
4. `/v1/chat/completions`는 더 자연스럽고 정돈됐지만, “1443 during the Goryeo Dynasty”처럼 명백한 역사 오류를 포함한다.
5. 따라서 긴 출력이 항상 더 지능적인 것은 아니고, 이 모델과 설정에서는 오히려 factual drift가 증가했다.

### 2-2-1. 실제로 관찰된 역사 오류 지적

아래는 테스트 응답에서 실제로 드러난 역사 오류 또는 역사적 환각 문장들이다.

#### `512 tokens` / `/v1/chat/completions`에서 나온 오류

1. `the first version of Hangul was created in 1443 during the Goryeo Dynasty in Korea.`
  문제점:
  Hangul 창제 시기는 1443년이 맞지만, 1443년은 고려가 아니라 조선 세종대이다. 고려는 1392년에 끝났기 때문에, `1443 during the Goryeo Dynasty`는 명백한 시대 착오다.

2. `King Sejong was a scholar and poet who played a crucial role in creating the Hangul alphabet ...`
  문제점:
  큰 틀에서는 맞는 설명이지만, 뒤 문맥과 이어지며 세부 사실을 부정확하게 섞어 쓰고 있다. 특히 아래의 연도 및 저작 관련 설명과 결합되면서 역사 정확도가 무너진다.

3. `He worked on the project from 1400 to 1450`
  문제점:
  세종은 1397년생이고, 세종 본인의 통치 시기와 훈민정음 창제 시점을 고려하면 `1400 to 1450`이라는 범위는 근거 없는 임의 확장이다. Hangul 창제는 일반적으로 1443 창제, 1446 반포로 설명하는 것이 맞다.

4. `his most famous work is the "Sejongmun" (Sejong's Dictionary), which was published in 1443.`
  문제점:
  `Sejongmun`이라는 대표 저작 명칭은 역사적으로 부정확하다. 세종과 Hangul 창제를 설명할 때 대표적으로 언급되는 문헌은 `훈민정음`이며, 반포 시점도 1446으로 보는 것이 일반적이다.

5. `MacBooks, as we know them today, were not invented until more than 600 years later, in the 1980s by Steve Jobs and Steve Wozniak at Apple Inc.`
  문제점:
  이 문장은 Apple 초기 컴퓨터 역사와 MacBook 제품군 역사를 혼동하고 있다. Apple 창업과 1980년대 Macintosh 계열, 그리고 `MacBook` 제품군 출시는 서로 다른 시점이다. `MacBook`이라는 제품명 자체는 훨씬 뒤 시기의 것이다.

#### `512 tokens` / `/generate`에서 나온 오류 또는 환각

1. `the first computers used by the Joseon Dynasty were likely to be simple machines like the "seungje" or "seongje"`
  문제점:
  조선 시대에 `컴퓨터`를 이런 식으로 연결하는 설명 자체가 시대적으로 부적절하고, `seungje`, `seongje`라는 표현도 역사적 근거가 불분명하다. 이는 사실 설명이 아니라 모델이 서사를 이어가며 만들어낸 환각에 가깝다.

2. `These early computers were likely to be based on mechanical or electromechanical systems`
  문제점:
  조선 시대 기술과 현대적 `computer` 개념을 무리하게 접목한 문장이다. 질문의 핵심은 단순한 역사적 반박인데, 길어진 출력에서 근거 없는 기술사 설명으로 drift한 것이다.

#### `32 tokens` 결과의 특성

1. `32 tokens` 결과는 길이가 짧아 큰 역사 오류가 본격적으로 드러나기 전에 잘렸다.
2. 따라서 짧은 출력에서는 치명적 오류가 덜 보였지만, 이는 모델이 더 정확해서라기보다 틀릴 기회가 줄어든 효과에 가깝다.

### 2-2-2. 왜 이런 역사 오류가 나왔는가

1. 출력 길이가 길어질수록 모델이 핵심 사실 반박 이후 주변 설명을 덧붙이며 drift했다.
2. `Llama-3.2-1B-Instruct` 규모에서는 일반 상식 질문에 대해 장문 응답을 할 때 factual consistency가 충분히 강하지 않았다.
3. system prompt가 도메인 한정 역할만 약하게 지정하고 있어서, 역사 질문에 대해 강하게 제어하지 못했다.
4. 그 결과 짧은 정답형 반박은 어느 정도 맞췄지만, 장문 설명 구간에서 시대, 저작, 기술사 정보가 섞이며 오류가 발생했다.

### 2-3. 지능평가 결론

1. 짧은 출력은 정확도는 상대적으로 낫지만 불완전하다.
2. 긴 출력은 표현은 좋아지지만 환각과 역사 오류가 증가한다.
3. 이 테스트에서 가장 큰 병목은 속도보다도 사실성 유지다.
4. 특히 `max_new_tokens`를 크게 열어둘수록 “더 많이 말하지만 더 정확하지는 않은” 경향이 보인다.

## 3. 두 번째 API에서 system prompt가 반영되지 않은 것처럼 보인 이유

결론부터 말하면, system prompt는 빠진 것이 아니라 실제로 들어갔다. 다만 system prompt의 내용이 약하고, 모델의 기본 일반지식 응답 성향이 더 강해서 결과적으로 system prompt가 체감상 먹지 않은 것처럼 보인 것이다.

### 3-1. 실제 system prompt 값

서버 코드에서 설정된 기본 system prompt는 다음 값이다.

```text
You are a domain assistant for marine DX physical AI and network systems.
```

이 값은 `jetson_slm_stack/app/server.py`에서 `SYSTEM_PROMPT` 환경변수가 없을 때 기본값으로 사용된다.

추적 결과:

1. 현재 `.env`에는 `SYSTEM_PROMPT=`가 명시되어 있지 않다.
2. 따라서 서버는 기본값을 사용한다.

### 3-2. chat API에서 실제로 system prompt가 삽입되는 경로

`/v1/chat/completions` 요청에서 `test_slm.sh`는 user message 하나만 보낸다.

개념적으로 전송 payload는 아래와 같다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Did King Sejong create Hangul while using a MacBook?"
    }
  ]
}
```

그 다음 서버에서 다음 일이 일어난다.

1. `build_prompt()`가 messages 안에 system role이 없음을 확인
2. 기본 system prompt를 맨 앞에 자동 삽입
3. tokenizer의 `apply_chat_template()`로 최종 문자열 프롬프트 생성
4. assistant 응답 시작 토큰을 붙여 generation 수행

즉 실제 내부 메시지 구조는 개념적으로 아래와 같다.

```json
[
  {
    "role": "system",
    "content": "You are a domain assistant for marine DX physical AI and network systems."
  },
  {
    "role": "user",
    "content": "Did King Sejong create Hangul while using a MacBook?"
  }
]
```

### 3-3. system prompt가 실제로 들어갔다는 증거

가장 직접적인 증거는 prompt token 증가다.

1. `/generate` prompt tokens: `13`
2. `/v1/chat/completions` prompt tokens: `62`

질문 본문은 동일한데 chat API의 prompt token이 훨씬 큰 이유는 다음이 합쳐졌기 때문이다.

1. system prompt 본문
2. `system`, `user`, `assistant` 헤더
3. tokenizer chat template의 special token들
4. 모델 템플릿 내부 메타데이터 문자열

즉 system prompt가 아예 누락됐다면 prompt token 차이가 이렇게 커지기 어렵다.

## 4. 왜 체감상 system prompt가 안 먹혔는가

### 원인 1. system prompt 자체가 너무 약하다

현재 문장은 단순히 다음만 말한다.

```text
You are a domain assistant for marine DX physical AI and network systems.
```

이 문장은 다음을 명시하지 않는다.

1. maritime/physical AI 외 질문은 거절하라
2. 역사 상식 질문에는 답하지 말라
3. 관련 없는 질문은 도메인 한정으로 되돌려라
4. 모르면 추정하지 말라

즉 “도메인 어시스턴트”라는 정체성만 부여할 뿐, 강한 제약 조건은 없다.

### 원인 2. chat template는 형식화 도구이지 강제 규칙이 아니다

Llama tokenizer의 chat template는 system message를 분명히 system 헤더 아래 넣는다. 하지만 template는 메시지를 구조화할 뿐이고, 모델이 그 지시를 반드시 따르게 만들지는 않는다.

즉 chat template의 역할은:

1. system / user / assistant 역할 구분
2. special token 삽입
3. assistant generation 시작 위치 지정

이지,

1. system prompt 준수 강제
2. 도메인 외 질문 차단

는 아니다.

### 원인 3. 모델의 기본 학습 성향이 더 강하다

`Llama-3.2-1B-Instruct`는 일반 instruction-following 모델이다. 따라서 user가 역사나 상식 질문을 하면, 약한 system prompt보다 “질문에 친절하게 답하려는 기본 성향”이 더 강하게 작동할 수 있다.

이번 결과가 바로 그 사례다.

1. 모델은 maritime assistant로 제한되기보다
2. 일반 지식 질문에 성실히 답하려고 했고
3. 출력 길이가 늘어나면서 사실 오류까지 늘어났다.

### 원인 4. 긴 generation에서 drift가 커진다

`512` 결과에서 특히 system prompt가 안 먹히는 것처럼 보이는 이유는, 길이가 길수록 모델이 원래 질문 바깥으로 drift하기 쉽기 때문이다.

실제 관찰:

1. `32 tokens`에서는 핵심 반박만 짧게 말하고 끝나므로 상대적으로 안정적
2. `512 tokens`에서는 설명을 계속 이어가며 사실 오류와 환각이 증가

즉 system prompt 미반영 체감은 단순 삽입 실패보다, 긴 generation에서의 instruction dilution과 factual drift 영향이 더 크다.

## 5. 최종 결론

1. 두 번째 API에서 system prompt는 실제로 빠진 것이 아니다.
2. 서버는 기본 system prompt를 자동 삽입하고, tokenizer chat template도 정상 적용하고 있다.
3. prompt token이 `13 -> 62`로 증가한 것이 그 간접 증거다.
4. 다만 현재 system prompt 문구가 너무 약해서, 모델 행동을 강하게 제한하지 못한다.
5. 따라서 결과적으로는 “system prompt가 반영되지 않은 것처럼 보이는” 응답이 나온다.
6. 특히 `max_new_tokens=512`처럼 긴 출력에서는 모델의 일반지식 응답 성향과 환각 경향이 더 강하게 드러난다.

## 6. 실무 해석

현재 상태를 실무적으로 해석하면 다음과 같다.

1. 시스템 프롬프트 삽입 로직 자체는 정상
2. 문제는 삽입 실패가 아니라 system prompt 설계 강도 부족
3. 도메인 강제를 원하면 더 강한 제약형 system prompt와 거절 규칙이 필요
4. 사실성 유지가 목적이면 긴 출력보다 짧은 출력이 더 안전
5. 현재 1B 모델은 도메인 강제와 사실성 유지에서 한계가 뚜렷하다