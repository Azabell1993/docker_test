# DGX 128 vs 512 Response Report

## 범위

이 문서는 동일한 질문에 대해 `max_new_tokens=128` 과 `max_new_tokens=512` 설정으로 실행한 성능 테스트 결과를 비교하고, 각 API 응답이 질문에 얼마나 적절했는지까지 평가한 리포트이다.

질문:

```text
How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?
```

공통 실행 조건:

1. Model: `meta-llama/Llama-3.2-1B-Instruct`
2. Device: `cuda`
3. Load mode: `cuda-offload-640mb`
4. dtype: `torch.float16`
5. temperature: `0.0`
6. top_p: `1.0`

## 1. 성능 비교 요약

### 1-1. 수치 비교 표

| Max New Tokens | API | Prompt Tokens | Completion Tokens | Total Tokens | Latency (sec) | Throughput (tok/s) | 관찰 |
|----------------|-----|---------------|-------------------|--------------|---------------|--------------------|------|
| 128 | `/generate` | 35 | 128 | 163 | 77.147 | 1.659 | 응답이 중간에 잘림 |
| 128 | `/v1/chat/completions` | 84 | 128 | 212 | 77.712 | 1.647 | assistant 형식이지만 중간에 잘림 |
| 512 | `/generate` | 35 | 512 | 547 | 439.407 | 1.165 | 끝까지 512 token을 사용, 매우 느림 |
| 512 | `/v1/chat/completions` | 84 | 512 | 596 | 334.057 | 1.533 | 구조는 좋지만 여전히 장문 생성 비용 큼 |

### 1-2. 성능 해석

1. `128` 설정은 두 API 모두 약 `77초` 수준으로 끝났다.
2. `512` 설정은 지연시간이 크게 증가했다.
3. 특히 `/generate`는 `77.147초 -> 439.407초`로 급격히 증가했다.
4. `/v1/chat/completions`도 `77.712초 -> 334.057초`로 크게 증가했다.
5. 처리량은 `128`에서 약 `1.65 tok/s`, `512`에서는 `/generate`가 `1.165 tok/s`까지 떨어졌다.
6. 즉 긴 출력이 된다고 해서 처리량이 좋아지지 않았고, 오히려 장시간 점유가 커졌다.

## 2. 응답 적절성 평가

평가 기준:

1. 질문 적합성: 사용자가 물은 DGX + Apache Spark + maritime + fine-tuning 파이프라인에 맞게 답했는가
2. 구체성: 실제 아키텍처 설계에 도움이 되는가
3. 완결성: 답변이 중간에 끊기지 않았는가
4. 정확성: 사실 오류나 근거 없는 일반화가 적은가
5. 실무 유용성: 바로 설계/구현 참고 자료로 쓸 수 있는가

### 2-1. 품질 비교 표

| Max New Tokens | API | 질문 적합성 | 구체성 | 완결성 | 정확성 | 실무 유용성 | 총평 |
|----------------|-----|-------------|--------|--------|--------|-------------|------|
| 128 | `/generate` | 보통 | 낮음 | 낮음 | 보통 | 낮음 | 일반적인 플랫폼 설명만 하다가 중간에서 종료 |
| 128 | `/v1/chat/completions` | 보통 이상 | 낮음 | 낮음 | 보통 | 낮음 | 형식은 더 좋지만 여전히 초반 개요 수준에서 잘림 |
| 512 | `/generate` | 보통 | 보통 이하 | 보통 | 보통 이하 | 낮음 | 말은 길지만 일반론과 플랫폼 나열 위주 |
| 512 | `/v1/chat/completions` | 보통 이상 | 보통 | 보통 | 보통 | 보통 이하 | 네 단계/다섯 단계 구조는 낫지만 여전히 구체 설계는 부족 |

## 3. 128 토큰 결과 평가

### 3-1. `/generate` 응답 평가

관찰:

1. 답변은 `Here's what I have so far:`로 시작하며 completion 스타일로 이어진다.
2. 초반에 `LLM Model`, `Model Serving`, `Apache Spark Integration`, `NVIDIA DGX Cluster`, `Physical AI Applications` 같은 상위 개념을 나열한다.
3. 하지만 실제 질문이 요구한 배포 및 파인튜닝 파이프라인 설계를 구체적으로 제시하기 전에 128 token 한도에서 잘린다.

적절성 판단:

1. 질문의 주제는 맞추고 있다.
2. 그러나 DGX 클러스터 구성, Spark 역할 분담, 데이터 파이프라인, 파인튜닝 루프, 모델 서빙 아키텍처 같은 핵심 설계 요소가 구체화되지 못했다.
3. 따라서 “완성된 답변”이라기보다 초안 수준이다.

결론:

1. 적절성은 부분적으로만 충족
2. 실무 참고용으로는 부족

### 3-2. `/v1/chat/completions` 응답 평가

관찰:

1. assistant 스타일로 `Step 1: Choose a suitable LLM model`부터 구조를 잡는다.
2. `/generate`보다 형식은 더 자연스럽다.
3. 그러나 128 token 제한 때문에 `Step 1` 일부만 설명하고 중간에서 끊긴다.

적절성 판단:

1. 질문을 더 잘 구조화해 시작했다는 점은 장점이다.
2. 하지만 여전히 실제 아키텍처 설계라고 부르기에는 내용이 부족하다.
3. maritime industry 맥락, DGX-Spark 통합 흐름, fine-tuning/serving 파이프라인 연결이 드러나기 전에 종료됐다.

결론:

1. `/generate`보다 형식은 낫다.
2. 하지만 128 token에서는 여전히 답변 완결성이 낮다.

## 4. 512 토큰 결과 평가

### 4-1. `/generate` 응답 평가

관찰:

1. `Here's what I have so far:`로 시작하며 일반적인 설명을 길게 이어간다.
2. `TensorFlow Serving`, `AWS SageMaker`, `Azure Machine Learning` 같은 여러 플랫폼을 예시로 들지만, 질문은 DGX + Spark 기반 아키텍처 설계를 요구하고 있다.
3. 특정한 DGX-Spark 파이프라인 설계안이라기보다 범용 LLM 배포 설명을 늘여놓는 형태에 가깝다.
4. `Use the TensorFlow Serving API ...` 같은 문장은 이 저장소의 Hugging Face / FastAPI 기반 스택과도 직접 연결되지 않는다.

적절성 판단:

1. 질문 키워드는 잡았지만 답변이 범용적이다.
2. 사용자가 원하는 것은 실제 아키텍처 설계인데, 답변은 플랫폼 예시 나열과 추상적 단계 설명 중심이다.
3. maritime industry용 데이터 소스, Spark ETL 역할, DGX 분산 학습 역할, 모델 registry, inference serving, evaluation loop 같은 설계 핵심이 빠져 있다.

결론:

1. 길어졌지만 충분히 적절하다고 보긴 어렵다.
2. 분량 증가가 곧 품질 향상으로 이어지지 않았다.

### 4-2. `/v1/chat/completions` 응답 평가

관찰:

1. 단계별 구조는 가장 잘 잡혀 있다.
2. `Choose a suitable LLM model`, `Prepare the data`, `Set up the NVIDIA DGX cluster`, `Integrate Apache Spark`, `Fine-tune the LLM`처럼 답변 형식은 가장 자연스럽다.
3. 하지만 여전히 내용은 상당히 일반론적이다.
4. `BERT`, `RoBERTa`, `XLNet` 제안은 “LLM 파이프라인”이라는 질문 맥락과 완전히 맞아떨어지지 않는다.
5. maritime Physical AI용 설계라면 멀티모달 데이터, telemetry ingestion, Spark batch/stream 처리, DGX 분산 학습, serving gateway, monitoring, feedback loop가 중심이 되어야 하는데 그 수준까지는 가지 못했다.

적절성 판단:

1. 네 API 결과 중에서는 가장 읽기 좋다.
2. 그러나 질문에 대한 정답형 설계 문서 수준은 아니다.
3. 실무적으로는 “방향 설명” 정도의 참고는 가능하지만, 바로 아키텍처 초안으로 쓰기에는 부족하다.

결론:

1. 네 결과 중 상대적으로 가장 적절한 응답
2. 그래도 구체성 부족 때문에 높은 평가를 주기는 어렵다.

## 5. 적절성 종합 평가

### 5-1. 어떤 응답이 더 적절했는가

순위를 매기면 다음과 같다.

1. `512` + `/v1/chat/completions`
2. `128` + `/v1/chat/completions`
3. `512` + `/generate`
4. `128` + `/generate`

이유:

1. Chat 경로가 항상 더 구조적이었다.
2. 512 token은 128보다 완결성이 높았다.
3. 하지만 512 `/generate`는 길어졌음에도 범용 플랫폼 나열이 많아 질문 적합성은 기대보다 낮았다.

### 5-2. 왜 완전히 적절하다고 보기 어려운가

질문이 요구한 것은 다음에 가깝다.

1. DGX 클러스터에서의 학습/추론 역할 분리
2. Apache Spark와의 데이터 처리 통합 방식
3. maritime Physical AI 도메인 데이터 파이프라인
4. 모델 파인튜닝, 배포, 평가, 모니터링을 잇는 end-to-end 아키텍처

하지만 실제 응답은 다음 한계가 있었다.

1. 일반적인 LLM 개론 수준에 머무름
2. 구체적인 컴포넌트 연결이 부족함
3. 실제 운영 아키텍처 대신 개념 나열에 가까움
4. DGX와 Spark를 어떻게 엮는지에 대한 설계 구체도가 낮음

## 6. 최종 결론

1. 성능 측면에서는 `128 tokens`가 훨씬 실용적이다.
2. `512 tokens`는 지연시간이 매우 커지지만, 그만큼 답변 품질이 비례해서 좋아지지는 않았다.
3. 응답 형식과 적절성은 `/v1/chat/completions`가 `/generate`보다 일관되게 낫다.
4. 그러나 네 결과 모두 “질문에 완전히 적절한 고품질 설계 답변”이라고 보기는 어렵다.
5. 현재 모델은 구조는 잡을 수 있지만, DGX + Spark + maritime Physical AI라는 복합 실무 아키텍처 질문에 대해 충분히 구체적이고 설계 가능한 수준의 답변을 안정적으로 주지는 못했다.

## 7. 실무 권장 해석

1. 빠른 데모나 smoke test에는 `128 tokens`가 적합
2. 더 읽기 좋은 응답 형식이 필요하면 `/v1/chat/completions`가 유리
3. 하지만 실제 설계 문서 초안이 필요하면 현재 1B 모델 출력만 신뢰하기 어렵다
4. 고품질 설계 답변이 목표라면 더 큰 모델, 더 강한 system prompt, 또는 retrieval/templated prompting이 필요하다