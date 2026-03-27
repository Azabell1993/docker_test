# Prepare Network Slicing Dataset 발표 Report

## 1. 발표 목적

이 보고서는 `prepare_network_slicing_dataset.py`가 어떤 문제를 해결하는지, CSV 원본 데이터가 어떤 과정을 거쳐 LLM 학습용 JSONL 데이터셋으로 바뀌는지, 그리고 그 과정에서 사용한 규칙과 설계 근거가 무엇인지를 발표용으로 설명하기 위해 작성되었다.

핵심 질문은 다음 세 가지다.

1. 원본 CSV에서 어떤 컬럼을 읽고 어떻게 정규화하는가
2. 왜 `instruction`, `output`, `metadata` 구조로 바꾸는가
3. 왜 `train/val/test = 8:1:1`과 같은 규칙 기반 설계를 사용했는가

## 2. 대상 코드와 역할

- 대상 파일: `jetson_slm_stack/dataset/scripts/prepare_network_slicing_dataset.py`
- 입력: `6G_network_slicing_qos_dataset_2345.csv`
- 출력:
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`
  - `manifest.prep.json`
  - `csv_schema.json`
  - `csv_preview.json`

이 스크립트는 단순 포맷 변환기가 아니다. CSV 한 행을 LLM instruction-tuning에 사용할 수 있는 supervised sample로 변환하는 데이터셋 생성기다.

즉 한 row에서 다음을 동시에 만든다.

- `instruction`: 모델에게 줄 문제 문장
- `output`: 규칙 기반으로 생성한 모범 답안
- `metadata`: 원본 row에서 선별한 구조화 정보

## 3. 전체 흐름

코드는 아래 5단계 흐름으로 동작한다.

### Flow 2. 대표 CSV 선택

- `detect_primary_csv()`가 우선 사용할 CSV를 선택한다.
이 설계는 파일명이 정확한 경우 재현성을 유지하고, 그렇지 않은 경우에도 파이프라인 전체가 중단되지 않도록 하기 위한 것이다.

### Flow 3. 헤더 inspection 및 컬럼 정규화 맵 생성

- `inspect_csv()`가 `csv.DictReader`로 헤더를 읽는다.
- `fieldnames`를 통해 원본 컬럼명을 확보한다.
- `slugify_column()`으로 컬럼명을 정규화한다.

예를 들면 다음과 같이 바뀐다.

- `Packet Loss Rate %` -> `packet_loss_rate_percent`

이 단계의 목적은 원본 CSV의 표기 흔들림을 제거하고, 이후 코드가 컬럼명을 일관되게 참조하도록 만드는 것이다.

### Flow 4. row 단위 JSONL 변환

- `convert_csv_to_jsonl()`가 CSV를 다시 열고 row를 한 줄씩 읽는다.
- 각 row마다 `choose_split()`으로 split을 결정한다.

### Flow 5. manifest, schema, preview 저장

- 전처리 결과 요약은 `manifest.prep.json`에 저장한다.
- 컬럼 구조는 `csv_schema.json`에 저장한다.
이 단계는 단순 부가 기능이 아니라 데이터 거버넌스 목적을 가진다. 즉, 어떤 CSV를 어떤 규칙으로 어떤 결과물로 바꿨는지 추적 가능하게 만든다.
## 4. 함수별 설명


역할:

- 문자열 값을 float으로 변환한다.
- 비어 있거나 잘못된 값은 `0.0`으로 대체한다.

- CSV 원본은 기본적으로 문자열이므로 수치 연산 전에 타입 정리가 필요하다.
- 데이터 품질이 완벽하지 않아도 파이프라인이 멈추지 않게 한다.


- 잘못된 값을 0으로 대체하면 오류를 숨길 수 있다.
- 운영 환경에서는 결측치와 이상치 로깅을 추가하는 것이 더 안전하다.

### 4.2 `coerce_int()`

역할:

의의:

- CSV에서 `1.0`처럼 들어오는 값을 `1`로 정리해 상태 플래그 비교가 가능하게 만든다.
- 이후 `if failure:` 같은 규칙 기반 분기에서 안정적으로 사용할 수 있다.

### 4.3 `write_json()` / `write_text()`
역할:

- 결과 파일을 저장한다.

의의:

- 전처리 산출물이 항상 같은 위치 구조에 떨어지도록 보장한다.
- 재실행 시 디렉터리 존재 여부 때문에 파이프라인이 실패하지 않게 한다.

### 4.4 `collect_raw_csvs()`


- raw 디렉터리에 존재하는 CSV 파일명을 전부 수집한다.
의의:

- `main()`의 Flow 1에 해당한다.

### 4.5 `slugify_column()`


- 원본 컬럼명을 코드 친화적인 key로 바꾼다.
변환 원리:

- 소문자화
- 공백을 `_`로 변경
- `%`를 `percent`로 변경
- `/`를 `_per_`로 변경
- 중복 `_` 제거


- CSV 공급자가 바뀌거나 컬럼 표기법이 조금 달라도 코드 수정 범위를 줄일 수 있다.

### 4.6 `detect_primary_csv()`



설계 철학:

- 기대 파일명이 있으면 명시적으로 선택한다.


- 엄격함과 실용성 사이에서 균형을 잡은 선택이다.

### 4.7 `inspect_csv()`

역할:
- 원본 컬럼 헤더를 읽고, 정규화 컬럼 맵을 만들고, 행 수와 preview를 계산한다.

왜 필요한가:

- 전처리 전에 데이터 구조를 먼저 이해해야 한다.


- `raw_file`
- `columns`
- `normalized_columns`
- `preview_rows`


역할:

- 각 row가 `train`, `val`, `test` 중 어디로 갈지 정한다.
구현 방식:

- `index % 10`을 계산한다.
- `0~7`은 train
- `8`은 val


- 이 숫자는 데이터셋이 스스로 제공한 정답이 아니다.
- 작성자가 실험 설계용 기본값으로 채택한 비율이다.

설득력 있는 근거는 다음과 같다.
1. 표본 효율
   - 대부분의 표본을 학습에 사용해야 모델이 패턴을 익힐 수 있다.
   - 따라서 `train = 80%`는 합리적인 기본값이다.
2. 검증 분리
   - 모델을 학습시키는 동안 설정과 품질을 확인할 독립 표본이 필요하다.
   - 그래서 `val = 10%`를 둔다.

3. 최종 평가 보존
   - 튜닝 과정에 직접 노출되지 않은 hold-out 데이터가 있어야 마지막 평가가 의미를 가진다.
   - 그래서 `test = 10%`를 둔다.

4. 재현성
   - 이 코드는 랜덤 분할이 아니라 deterministic 분할이다.
   - 발표, 비교 실험, 디버깅에 매우 유리하다.

split은 무엇으로 쓰이는가:

- 각 row를 어떤 출력 파일에 저장할지 결정하는 라우팅 기준이다.
- 즉 `train`, `val`, `test`는 각각 `train.jsonl`, `val.jsonl`, `test.jsonl`로 이어진다.

한계:

- row 순서가 시간순, 지역순, 장비순으로 정렬되어 있다면 편향이 split 전체로 전파될 수 있다.

### 4.9 `build_instruction()`

역할:



- 따라서 CSV 한 행을 네트워크 엔지니어링 상황 설명으로 바꿔주는 중간 계층이 필요하다.

설계 포인트:
- 고정 템플릿 사용
- 핵심 필드를 bullet로 나열
- float는 소수점 6자리로 고정

효과:
- 프롬프트 형식의 일관성 확보
- 샘플 간 비교 용이성 증가

### 4.10 `classify_qos_state()`
역할:

- metadata를 바탕으로 `stable`, `degraded`, `critical` 중 하나를 부여한다.

중요한 해석:
- 이 함수는 예측 모델이 아니다.
- 규칙 기반 weak labeling 엔진이다.
- 즉, 원본 데이터에 정답 라벨이 없거나 부족한 상황에서 전문가 판단을 흉내 낸 기준 라벨을 생성한다.


- `critical`
  - `overload_status == 1`
  - `latency >= 0.65`
  - `packet_loss >= 0.65`

  - `latency >= 0.45`
  - `packet_loss >= 0.40`
  - `throughput <= 0.35`

- 그 외는 `stable`
왜 이런 숫자인가:
- 현재 코드만 놓고 보면 이 값들은 통계 추정 결과가 아니라 휴리스틱 threshold다.
- 작성자는 latency와 packet loss가 높고 throughput이 낮을수록 QoS 품질이 나빠진다는 도메인 직관을 수치 경계값으로 고정했다.
- 또한 실제 장애와 과부하 플래그는 수치보다 우선하는 hard signal로 취급했다.

주의점:

- 만약 실제 단위가 절대 ms, 절대 %, 절대 bps라면 threshold는 재설계해야 한다.

### 4.11 `recommend_action()`



우선순위:

- 장애
- 지연
- 손실
- 안정 상태

의의:

- QoS 상태 분류만으로 끝내지 않고, 실제 운영자 대응 문장까지 생성한다.
- 따라서 단순 분류 데이터셋이 아니라 설명형 instruction 데이터셋으로 확장된다.

### 4.12 `build_expected_output()`


- `instruction`에 대한 모범 답안을 생성한다.

구성:

2. Failure risk

왜 중요한가:

- 이 함수 덕분에 CSV row가 supervised fine-tuning용 `(instruction, output)` 샘플로 바뀐다.
- 즉 이 스크립트의 가장 큰 의미는 표 데이터를 바로 LLM 튜닝용 데이터로 바꾸는 데 있다.
`failure_risk` 기준:

  - 실제 장애 플래그 존재
  - 과부하 존재
  - `network_failure_count >= 2`

- `medium`
  - latency나 packet loss가 경계값 이상

  - 나머지

해석:

- 따라서 ground truth 정답이라기보다 전문가 판단을 모사한 synthetic supervision에 가깝다.

### 4.13 `normalize_row()`

역할:
- 원본 row의 key를 정규화 컬럼명으로 바꾼다.


- 이 함수가 실질적인 row-level normalization 수행 지점이다.
- 이후 로직은 모두 정규화된 key만 보고 동작한다.

### 4.14 `build_record()`


- JSONL 한 샘플의 핵심 payload를 만든다.

처리 순서:

2. 필요한 컬럼만 선별
4. `instruction` 생성
5. `output` 생성
6. 최종 payload 반환

중요 포인트:

- 이 함수가 사실상 전처리 스키마 정의 지점이다.
- 원본 CSV의 모든 컬럼을 쓰지 않고, 의미 있는 컬럼만 선택한다.

즉 발표에서 다음처럼 설명할 수 있다.
- 원본 CSV row 전체를 그대로 쓰지 않는다.
- LLM 판단에 필요한 최소 구조만 `metadata`로 남긴다.

### 4.15 `convert_csv_to_jsonl()`


- 실제 파일 변환 루프를 담당한다.

무슨 일이 일어나는가:

1. split별 출력 파일 핸들을 연다.
2. CSV를 다시 읽는다.
4. `build_record()`로 payload를 만든다.
5. 해당 split JSONL 파일에 한 줄씩 저장한다.
여기서 split은 무엇으로 쓰이는가:

- 단순 라벨이 아니다.
- 최종적으로 어떤 출력 파일에 저장될지를 결정하는 라우팅 기준이다.
- 즉 `train`, `val`, `test`는 곧 `train.jsonl`, `val.jsonl`, `test.jsonl` 파일 선택 규칙이다.




순서:

1. 디렉터리 준비
2. raw CSV 목록 수집
3. 대표 CSV 선택
5. row 단위 JSONL 변환
7. 로그 출력
발표에서는 `main()`을 전체 시스템 orchestration 계층으로 설명하면 된다.

## 5. 이 코드의 학술적 및 실무적 의미

이 스크립트의 가치는 단순 CSV 전처리에 있지 않다. 더 중요한 의미는 다음과 같다.


`classify_qos_state()`와 `build_expected_output()`은 정답 라벨이 완비되지 않은 상황에서 전문가 규칙으로 surrogate label을 만든다. 이는 약지도 데이터셋 구축의 전형적 전략이다.

랜덤 분할 대신 deterministic split을 사용한 것은 연구 재현성과 디버깅 편의성을 높인다. 특히 발표, 보고서, 후속 실험 비교에서 같은 샘플 구성을 반복 재현할 수 있다는 점이 중요하다.

## 6. 한계와 보완 방향
### 6.1 threshold의 경험적 성격

- 현재 threshold는 경험적 heuristic이다.
- 원본 데이터 분포를 기반으로 최적화된 값은 아니다.

보완 방향:

- 컬럼 분포 통계 산출
- percentile 기반 threshold 재설정
- 도메인 전문가 검토

### 6.2 split 방식의 순서 의존성

- index 기반 split은 재현성이 높지만 row 순서 편향에 취약하다.

보완 방향:

- random split
- stratified split
- time-based split

### 6.3 결측치 처리의 보수성

- 현재는 오류 대신 `0` 또는 빈 문자열로 대체한다.
- 이는 파이프라인 안정성은 높이지만 silent degradation을 유발할 수 있다.

보완 방향:

- 필수 컬럼 검증
- 결측치 로깅
- 비정상 row quarantine

## 7. 핵심 포인트
1. 이 코드는 CSV를 단순 변환하는 것이 아니라, 네트워크 QoS snapshot을 LLM instruction-tuning 샘플로 재구성한다.
2. 컬럼명 정규화는 원본 스키마의 흔들림을 제거해 이후 로직의 일관성을 보장한다.
3. `build_record()`는 전처리 파이프라인의 핵심으로, 필요한 컬럼만 골라 metadata를 만들고 그로부터 instruction과 output을 생성한다.
4. `classify_qos_state()`와 `build_expected_output()`은 규칙 기반 약지도 엔진으로서 synthetic supervision을 만든다.
5. `8:1:1` split은 최적 해를 증명한 숫자가 아니라, 학습 효율, 검증 분리, 최종 평가 보존, 재현성을 동시에 만족시키는 실용적 기본값이다.

## 8. 결론

`prepare_network_slicing_dataset.py`는 네트워크 QoS CSV를 LLM 평가 및 튜닝용 JSONL 데이터셋으로 바꾸는 핵심 준비 단계다. 이 스크립트는 다음 세 가지를 동시에 달성한다.

1. 컬럼 스키마 정규화
2. 규칙 기반 라벨 및 설명 생성
3. 재현 가능한 split과 산출물 관리

따라서 이 코드는 단순 전처리 스크립트가 아니라, 정형 네트워크 운영 데이터를 설명형 LLM 실험 데이터셋으로 전환하는 데이터셋 설계 계층이라고 평가할 수 있다.