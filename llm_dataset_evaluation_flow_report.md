# LLM Dataset Evaluation Flow Report

## 목적

- `test_dataset.sh`가 `test.jsonl`의 각 샘플을 어떻게 읽고
- LLM 서버에 어떤 형태로 요청을 보내며
- 생성된 응답을 어떤 JSON 구조로 `dataset_eval_*.jsonl`에 저장하는지 정리

## 핵심 흐름

1. 입력 데이터셋 로드
- 파일: `jetson_slm_stack/dataset/prepared/network_slicing_qos/test.jsonl`
- `test_dataset.sh`는 split 파일 경로를 설정한 뒤 전체 줄 수를 계산
- 관련 코드:
  - `API_PATH` 설정: `test_dataset.sh:133`
  - 서버 health check: `test_dataset.sh:143-150`
  - 샘플 수 계산: `test_dataset.sh:159-163`

2. JSONL 한 줄씩 읽기
- 메인 평가 루프에서 `while IFS= read -r line; do`로 한 줄씩 읽음
- 각 줄에서 `id`, `instruction`, `input`, `output(expected)`를 `jq`로 추출
- 관련 코드:
  - 루프 시작: `test_dataset.sh:204`
  - 필드 추출: `test_dataset.sh:211-214`

3. LLM 프롬프트 구성
- `instruction`을 기본 프롬프트로 사용
- `input`이 있으면 아래에 추가로 이어 붙임
- 현재 네트워크 슬라이싱 데이터셋은 대부분 `input`이 빈 문자열이므로, 실제 요청 프롬프트의 중심은 `instruction`
- 관련 코드:
  - `FULL_PROMPT` 구성: `test_dataset.sh:216-221`

4. LLM 서버 호출

### chat API 경로

- `--api chat`인 경우 `/v1/chat/completions` 호출
- 요청 JSON은 아래 구조로 만들어짐

```json
{
  "messages": [
    {
      "role": "user",
      "content": "instruction 기반 프롬프트"
    }
  ],
  "max_new_tokens": 128,
  "temperature": 0.0,
  "top_p": 0.9
}
```

- 서버 응답에서는 아래 값을 꺼냄
  - `.choices[0].message.content` → `generated`
  - `.usage.prompt_tokens`
  - `.usage.completion_tokens`
  - `.latency_sec`
  - `.tokens_per_sec`
- 관련 코드:
  - chat payload 생성: `test_dataset.sh:227-236`
  - chat 요청 전송: `test_dataset.sh:237-239`
  - chat 응답 파싱: `test_dataset.sh:240-244`

### generate API 경로

- `--api generate`인 경우 `/generate` 호출
- 관련 코드:
  - generate payload 생성: `test_dataset.sh:246-254`
  - generate 요청 전송: `test_dataset.sh:255-257`
  - generate 응답 파싱: `test_dataset.sh:258-262`

5. 실패 처리
- `generated`가 비어 있거나 `null`이면 실패 샘플로 기록
- 이 경우 결과 파일에는 `id`, `status=fail`, `error`만 저장
- 관련 코드:
  - 실패 판정 및 저장: `test_dataset.sh:265-273`

6. 성공 응답 저장
- LLM 응답이 정상적으로 오면 아래 JSON 구조로 결과 파일에 한 줄씩 append
- 관련 코드:
  - 성공 결과 저장: `test_dataset.sh:286-307`

```json
{
  "id": "network-slicing-00009",
  "status": "ok",
  "instruction": "LLM에게 보낸 실제 분석 프롬프트",
  "expected": "규칙 기반으로 생성한 기준 답변",
  "generated": "LLM이 실제로 생성한 응답",
  "metrics": {
    "prompt_tokens": 271,
    "completion_tokens": 128,
    "latency_sec": 375.473,
    "tokens_per_sec": 0.341
  }
}
```

## `dataset_eval_llama_test_20260324_193144.jsonl` 해석

### 입력 JSONL 샘플의 의미

- `instruction`
  - CSV 한 행에서 추출한 QoS 수치들을 자연어 프롬프트로 바꾼 값
  - 실제로 Llama에 전달되는 핵심 입력
- `expected`
  - 데이터셋 준비 스크립트가 규칙 기반으로 만든 기준 답변
  - 정답 레이블이라기보다 "비교 기준" 역할
- `generated`
  - Llama 서버가 `/v1/chat/completions` 또는 `/generate`를 통해 실제 생성한 응답
- `metrics`
  - 토큰 수, 지연 시간, 초당 토큰 수 등 성능 지표

### 예시 1: `network-slicing-00009`

- 입력 특징:
  - `latency_ms = 0.736525`
  - `packet_loss_rate_percent = 0.391512`
  - `qos_metric_throughput = 0.980159`
  - `network_failure_count = 4`
- 기준 답변(`expected`):
  - QoS state = `critical`
  - Failure risk = `high`
  - Action = 저지연 트래픽 우선화
- 실제 Llama 응답(`generated`):
  - QoS state를 `OK`로 판단
  - Failure risk를 `Low`로 판단

해석:
- 이 샘플에서 Llama는 `throughput`를 긍정적으로 과대 해석했고
- `latency`와 `network_failure_count` 같은 위험 신호를 충분히 반영하지 못함
- 즉, `generated`는 기준 규칙보다 더 낙관적인 방향으로 응답함

### 예시 2: `network-slicing-00019`

- 입력 특징:
  - `packet_loss_rate_percent = 0.938765`
  - `overload_status = 1`
  - `network_slice_failure = 1`
  - `network_failure_count = 4`
- 기준 답변(`expected`):
  - QoS state = `critical`
  - Failure risk = `high`
  - Action = slice recovery 권고
- 실제 Llama 응답(`generated`):
  - Failure risk는 `High`로 맞췄지만
  - QoS state는 `Optimized`로 표현

해석:
- 이 샘플에서는 장애/실패 관련 이산 변수는 일부 반영했지만
- QoS 상태 설명은 실제 위험 수준보다 부정확하게 요약함

## 전체 구조 요약

```text
CSV 원본 행
-> prepare_network_slicing_dataset.py가 instruction / expected / metadata 생성
-> train.jsonl / val.jsonl / test.jsonl 저장
-> test_dataset.sh가 test.jsonl을 한 줄씩 읽음
-> instruction을 LLM 서버에 전송
-> LLM 응답을 generated로 수신
-> expected와 generated를 함께 dataset_eval_*.jsonl에 저장
```

## 결론

- `test_dataset.sh`는 단순 파일 변환 스크립트가 아니라 LLM 자동 평가 실행기임
- `dataset_eval_*.jsonl`의 `generated`는 규칙 기반 출력이 아니라 실제 Llama 응답임
- 따라서 이 결과 파일은
  - 데이터셋 입력
  - 기준 답변
  - 실제 LLM 응답
  - 성능 지표
를 한 줄 단위로 묶어 저장한 평가 로그 역할을 함