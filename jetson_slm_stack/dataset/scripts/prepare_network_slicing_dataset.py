from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


# 현재 스크립트 파일 기준으로 raw/prepared 데이터셋 경로를 계산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = DATASET_DIR / "raw" / "network_slicing_qos"
DEFAULT_PREP_ROOT = DATASET_DIR / "prepared" / "network_slicing_qos"

RAW_ROOT = Path(os.getenv("DATASET_RAW_DIR", str(DEFAULT_RAW_ROOT)))
PREP_ROOT = Path(os.getenv("DATASET_PREP_DIR", str(DEFAULT_PREP_ROOT)))
EXPECTED_RAW_FILE = "6G_network_slicing_qos_dataset_2345.csv"

# 데이터셋 메타 정보다.
# 전처리 결과 manifest.prep.json에 그대로 포함되어, 어떤 데이터셋을 어떤 의도로 준비했는지 설명하는 용도다.
DATASET_SPEC = {
    "name": "6G Network Slicing QoS",
    "name_ko": "네트워크 슬라이싱",
    "source": {
        "provider": "Kaggle",
        "url": "https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset",
        "dataset_id": "ziya07/wireless-network-slicing-dataset",
    },
    "status": "prep-only",
    "format": "csv",
    "characteristics": [
        "throughput",
        "latency",
        "packet_loss_rate",
    ],
    "primary_uses": [
        "QoS prediction",
        "slice optimization",
    ],
    "llm_use_cases": [
        "지연에 민감한 슬라이스 분석",
        "QoS 이상 탐지 질의",
        "슬라이스 정책 변경 전후 비교 설명",
    ],
    "expected_raw_files": [
        EXPECTED_RAW_FILE,
    ],
    "normalization_plan": {
        "target_schema": [
            "slice_id",
            "service_type",
            "throughput",
            "latency",
            "packet_loss_rate",
            "timestamp",
            "label",
        ],
        "pending_tasks": [
            "inspect original Kaggle columns",
            "map raw CSV columns to target schema",
            "define train/val/test split strategy",
            "design instruction-answer templates for LLM anomaly analysis",
        ],
    },
}


# train/val/test 분할 규칙이다.
# 여기서 8:1:1은 데이터셋 자체가 요구한 비율이 아니라, 전처리 작성자가 실험 설계 관점에서 채택한 기본값이다.
# 이 비율을 택한 이유는 다음과 같다.
# 1) 학습 데이터 효율: 전체 표본의 대부분(80%)을 train에 두어 모델이 패턴을 학습할 충분한 표본을 확보한다.
# 2) 검증 데이터 분리: val 10%를 따로 두어 학습 중 프롬프트/하이퍼파라미터/모델 선택을 점검할 수 있게 한다.
# 3) 최종 평가 보존: test 10%를 별도로 남겨, 튜닝에 직접 사용하지 않은 hold-out 평가를 유지한다.
# 4) 실무적 균형: 80/10/10은 표본 효율과 평가 신뢰도 사이에서 널리 쓰이는 보수적 기본 분할이다.
# 즉 "최적으로 증명된 숫자"라기보다, 작은 전처리 파이프라인에서 재현 가능하고 해석 가능한 출발점으로 선택한 값이다.
# 또한 현재 코드는 무작위 샘플링이 아니라 index 기반 deterministic split을 사용하므로,
# 같은 CSV에 대해 항상 같은 결과를 재생산할 수 있다는 장점이 있다.
SPLIT_RULE = {"train": 8, "val": 1, "test": 1}


# CSV에서 읽은 값은 기본적으로 문자열이므로 float 컬럼을 안전하게 변환한다.
# 값이 비어 있거나 형식이 잘못된 경우에도 파이프라인이 멈추지 않게 0.0으로 대체한다.
def coerce_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# 분류 코드, 상태 플래그, count 컬럼은 int로 맞춘다.
# 예를 들어 "1.0"처럼 들어와도 int(float(...))로 정수화한다.
def coerce_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# README 같은 일반 텍스트 파일을 저장하는 유틸리티다.
def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# [Flow 1] raw 폴더 안의 CSV 파일명을 전부 수집한다.
# 실제 main에서 가장 먼저 수행되는 데이터 탐색 단계다.
def collect_raw_csvs() -> list[str]:
    return sorted(path.name for path in RAW_ROOT.glob("*.csv"))


# 원본 CSV 헤더를 코드에서 일관되게 쓰기 위한 snake_case 형태로 정규화한다.
# 예:
# - "Latency(ms)" -> "latency_ms"
# - "Packet Loss Rate %" -> "packet_loss_rate_percent"
# 이렇게 바꿔두면 이후 row 처리에서 컬럼명이 달라도 접근 규칙을 일정하게 유지할 수 있다.
def slugify_column(name: str) -> str:
    value = name.strip().lower()

    for source, target in (("%", "percent"), ("/", "_per_"), ("-", "_"), (" ", "_")):
        value = value.replace(source, target)

    value = re.sub(r"[^a-z0-9_]", "_", value)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


# [Flow 2] 어떤 CSV를 실제 변환 대상으로 쓸지 결정한다.
def detect_primary_csv(raw_files: list[str]) -> Path | None:
    # 기대한 파일명이 있으면 그 파일을 우선 사용한다.
    expected = RAW_ROOT / EXPECTED_RAW_FILE
    if expected.exists():
        return expected
    # 기대한 파일이 없더라도 CSV가 하나라도 있으면 첫 번째 파일을 fallback으로 사용한다.
    if raw_files:
        return RAW_ROOT / raw_files[0]
    return None


# [Flow 3] 1차 inspection 단계:
# - CSV 헤더(fieldnames)를 읽어 원본 메타데이터 컬럼 목록을 확보하고
# - 각 컬럼명을 slugify_column으로 정규화한 매핑을 만든다.
# 이 normalized_columns가 이후 실제 row 변환 시 기준이 된다.
def inspect_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        
        # DictReader가 첫 줄 헤더를 읽어 컬럼명을 fieldnames로 제공한다.
        # 질문하신 "메타데이터 컬럼을 어디서 읽느냐"의 시작점이 바로 여기다.
        fieldnames = reader.fieldnames or []

        # 원본 컬럼명 -> 정규화 컬럼명 매핑을 만든다.
        normalized_columns = {column: slugify_column(column) for column in fieldnames}

        preview_rows: list[dict[str, str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(preview_rows) < 5:
                # preview도 정규화된 컬럼명 기준으로 저장해 이후 schema 확인을 쉽게 한다.
                preview_rows.append({normalized_columns[key]: value for key, value in row.items() if key is not None})

    return {
        "raw_file": path.name,
        "row_count": row_count,
        "columns": fieldnames,
        "normalized_columns": normalized_columns,
        "preview_rows": preview_rows,
    }


# row index를 기준으로 deterministic split을 고른다.
# split은 각 row가 어떤 목적의 데이터로 쓰일지를 정하는 레이블이다.
# - train: 모델이 학습할 샘플
# - val: 학습 중간에 설정/프롬프트/성능을 점검할 샘플
# - test: 최종 성능을 별도로 평가할 샘플

# 구현 방식은 index % 10을 사용한 주기적 배치다.
# 따라서 10개 row마다 앞의 8개는 train, 다음 1개는 val, 마지막 1개는 test로 간다.
# 이 방식의 핵심 목적은 무작위성보다 재현성을 우선하는 것이다.

# 즉 같은 입력 CSV와 같은 row 순서가 주어지면 split 결과도 항상 동일하다.
# 다만 이 접근은 CSV row 순서가 이미 편향되어 있지 않다는 가정을 둔다.
def choose_split(index: int) -> str:
    bucket = index % 10
    if bucket < SPLIT_RULE["train"]:
        return "train"
    if bucket < SPLIT_RULE["train"] + SPLIT_RULE["val"]:
        return "val"
    return "test"

# metadata record를 사람이 읽는 프롬프트 문자열로 바꾼다.
# 즉 build_instruction은 "모델에게 무엇을 하라고 시킬지"를 만드는 함수다.
# 여기서 metadata의 핵심 수치와 코드값을 bullet 형식으로 나열해
# LLM이 한 샘플의 네트워크 상태를 요약 판단할 수 있게 만든다.
# 목적:
# - 구조화된 수치/범주형 필드를 자연어 프롬프트로 변환
# - 모델이 단일 snapshot 기준으로 QoS 상태, 장애 위험, 즉시 조치를 판단하도록 유도
#
# 표현 방식:
# - bullet 형태로 각 필드를 고정 순서로 나열
# - float 값은 소수점 6자리까지 고정 출력하여 문자열 흔들림 최소화
def build_instruction(record: dict[str, str]) -> str:
    return (
        "You are assisting a 6G network slicing QoS engineer. "
        "Analyze the following network slice snapshot and return a compact assessment with "
        "1) QoS state, 2) failure risk, and 3) immediate optimization action.\n\n"
        f"- slice_id: {record['network_slice_id']}\n"
        f"- timestamp: {record['timestamp']}\n"
        f"- device_id: {record['device_id']}\n"
        f"- traffic_load_bps: {record['traffic_load_bps']:.6f}\n"
        f"- traffic_type_code: {record['traffic_type']}\n"
        f"- network_utilization_percent: {record['network_utilization_percent']:.6f}\n"
        f"- latency_ms: {record['latency_ms']:.6f}\n"
        f"- packet_loss_rate_percent: {record['packet_loss_rate_percent']:.6f}\n"
        f"- signal_strength_dbm: {record['signal_strength_dbm']:.6f}\n"
        f"- bandwidth_utilization_percent: {record['bandwidth_utilization_percent']:.6f}\n"
        f"- throughput_qos_metric: {record['qos_metric_throughput']:.6f}\n"
        f"- overload_status: {record['overload_status']}\n"
        f"- network_slice_failure: {record['network_slice_failure']}\n"
        f"- network_failure_count: {record['network_failure_count']}\n"
        f"- device_type_code: {record['device_type']}\n"
        f"- region_code: {record['region']}\n"
        f"- time_of_day_code: {record['time_of_day']}\n"
        f"- weather_conditions_code: {record['weather_conditions']}"
    )


# metadata record를 규칙 기반으로 stable / degraded / critical 셋 중 하나로 분류한다.
# 여기의 숫자 기준은 이 코드 안에서 학습된 값이 아니라, 작성자가 수동으로 정한 heuristic이다.
# 전제는 latency, packet_loss, throughput이 0~1 또는 그에 준하는 정규화 스케일이라는 것이다.

# 기준값 의미:
# - 0.65 이상 latency 또는 packet_loss: 즉시 심각 상태로 간주
# - 0.45 이상 latency, 0.40 이상 packet_loss, 0.35 이하 throughput: 성능 저하로 간주
# - overload_status / network_slice_failure가 1이면 수치와 무관하게 더 우선해서 위험 상태로 본다.
# 따라서 이 함수는 "모델 정답"을 만드는 규칙 엔진이며, 통계 추정이 아니라 도메인 휴리스틱 기반 라벨러다.
#
# 중요한 점:
# - 이 함수는 "학습된 모델"이 아니라 "정답 유사 라벨을 생성하는 rule engine"이다.
# - 즉 output 생성용 weak labeling 로직에 가깝다.
#
# 현재 전제:
# - latency_ms, packet_loss_rate_percent, qos_metric_throughput이
#   이미 비교 가능한 스케일(예: 0~1 또는 그에 준하는 normalized range)에 있다고 가정한다.
#
# 리스크:
# - 실제 원본 값이 ms/%/bps 절대값이라면 현재 threshold는 무의미해질 수 있다.
# - 따라서 inspect 결과와 원본 통계분포를 확인한 뒤 threshold 타당성을 재검증해야 한다.
def classify_qos_state(record: dict[str, str | int | float]) -> str:
    latency = float(record["latency_ms"])
    packet_loss = float(record["packet_loss_rate_percent"])
    throughput = float(record["qos_metric_throughput"])
    overload = int(record["overload_status"])
    failure = int(record["network_slice_failure"])

    # 장애 플래그가 있거나 latency / packet_loss가 매우 높으면 critical로 올린다.
    if failure or overload or latency >= 0.65 or packet_loss >= 0.65:
        return "critical"
    
    # critical까진 아니지만 품질이 떨어졌다고 볼 수 있는 경계값이다.
    if latency >= 0.45 or packet_loss >= 0.40 or throughput <= 0.35:
        return "degraded"
    
    # 위 조건에 걸리지 않으면 상대적으로 안정적이라고 본다.
    return "stable"


# QoS 상태와 원인 지표를 바탕으로 즉시 취할 액션 문장을 만든다.
# 우선순위는 장애 -> 과부하 -> 지연 -> 손실 -> 낮은 처리량 -> 안정 상태 순이다.
# 즉 여러 조건이 동시에 참이어도 앞 조건이 더 중요하다고 가정하고 먼저 반환한다.
def recommend_action(record: dict[str, str | int | float], qos_state: str) -> str:
    overload = int(record["overload_status"])
    failure = int(record["network_slice_failure"])
    latency = float(record["latency_ms"])
    packet_loss = float(record["packet_loss_rate_percent"])
    throughput = float(record["qos_metric_throughput"])

    # 실제 장애 플래그가 있으면 복구와 격리가 최우선이다.
    if failure:
        return "trigger slice recovery, inspect the affected device path, and isolate unstable network functions"
    
    # 장애는 아니지만 과부하라면 트래픽 재분산이 최우선이다.
    if overload:
        return "rebalance traffic across slices and reduce overload on the current slice immediately"
    
    # 지연이 높으면 스케줄링과 우선순위 정책을 조정한다.
    if latency >= 0.45:
        return "prioritize low-latency traffic and tighten scheduling for latency-sensitive flows"
    
    # 패킷 손실이 높으면 신뢰성 보강이 우선이다.
    if packet_loss >= 0.40:
        return "investigate loss sources and increase reliability controls before more traffic is admitted"
    
    # 처리량 지표가 낮으면 대역폭 재할당 쪽을 추천한다.
    if throughput <= 0.35:
        return "optimize throughput allocation and reclaim underused bandwidth from neighboring slices"
    
    # stable이면 큰 정책 변화 대신 모니터링 유지로 끝낸다.
    if qos_state == "stable":
        return "keep the current slice policy and continue telemetry monitoring for drift"
    
    # 나머지는 보수적 튜닝이라는 일반 대응 문장으로 처리한다.
    return "review current slice policy and tune resource allocation conservatively"


# build_expected_output은 instruction에 대한 모범 답안 역할을 하는 정답 문자열을 만든다.
# 즉 이 스크립트는 CSV 한 row에서
# - instruction: 모델에게 줄 문제
# - output: 그 문제의 기대 답안
# 을 동시에 생성한다.
# 출력 구성:
# 1) QoS state
# 2) Failure risk
# 3) Recommended action
#
# 이 함수의 역할:
# - 단순 전처리를 넘어, CSV row를 instruction-tuning용 supervised sample로 변환
# - 즉 한 row에서 (문제, 모범답안)을 동시에 생성한다.
#
# 주의:
# - failure_risk 역시 heuristic이며, 실측 정답 라벨이 아니다.
# - 따라서 이 데이터는 "전문가 판단 시뮬레이션용 synthetic supervision"으로 보는 편이 정확하다.
def build_expected_output(record: dict[str, str | int | float]) -> str:

    # QoS 상태는 classify_qos_state의 규칙 기반 판단을 그대로 사용한다.
    qos_state = classify_qos_state(record)

    # failure_risk는 QoS 상태와 별개로 위험도를 high/medium/low로 한 번 더 축약한 값이다.
    # 기준 역시 heuristic이며 다음 우선순위를 따른다.
    # - high: 실제 장애 플래그, 과부하, 혹은 failure_count >= 2
    # - medium: latency / packet_loss가 경계값 이상
    # - low: 나머지
    failure_risk = "high" if int(record["network_slice_failure"]) or int(record["overload_status"]) or int(record["network_failure_count"]) >= 2 else "medium" if float(record["latency_ms"]) >= 0.45 or float(record["packet_loss_rate_percent"]) >= 0.40 else "low"

    # 액션 추천은 별도 함수로 만들어 재사용한다.
    action = recommend_action(record, qos_state)
    return (
        f"1) QoS state: {qos_state} because latency={float(record['latency_ms']):.3f}, "
        f"packet_loss={float(record['packet_loss_rate_percent']):.3f}, throughput={float(record['qos_metric_throughput']):.3f}. "
        f"2) Failure risk: {failure_risk} because slice_failure={int(record['network_slice_failure'])}, "
        f"overload={int(record['overload_status'])}, network_failure_count={int(record['network_failure_count'])}. "
        f"3) Recommended action: {action}."
    )


# 실제 row 한 줄에 normalized_columns 매핑을 적용해
# 원본 CSV key를 정규화된 key로 바꾸는 단계다.
def normalize_row(row: dict[str, str], normalized_columns: dict[str, str]) -> dict[str, str]:
    return {normalized_columns[key]: value for key, value in row.items() if key is not None}


# [Flow 4-1] JSONL 한 샘플을 만드는 핵심 단계다.
#
# 처리 단계:
# 1) normalize_row()로 원본 컬럼명을 canonical key로 변환
# 2) 필요한 컬럼만 선별해 metadata(record) 구성
# 3) record 기반으로 instruction / output 생성
# 4) 학습 샘플 payload 형태로 반환
#
# 설계 포인트:
# - 전체 CSV 컬럼을 그대로 싣지 않고, 모델 입력/판단에 필요한 필드만 유지
# - 따라서 이 함수가 사실상 "학습용 스키마 정의 지점"이다
#
# 주의:
# - 없는 컬럼은 0 또는 빈 문자열로 대체되므로 파이프라인은 유지되지만,
#   조용한 데이터 품질 저하(silent degradation)가 발생할 수 있다.
# - 운영용이라면 필수 컬럼 누락 검증을 별도로 두는 편이 안전하다.
def build_record(index: int, row: dict[str, str], normalized_columns: dict[str, str]) -> dict:
    # normalize는 여기서 실제 변환 row에 적용된다.
    normalized = normalize_row(row, normalized_columns)

    # 이 record가 최종 JSONL의 metadata가 된다.
    # 전체 CSV 컬럼을 다 쓰지 않고, 모델 입력과 판단에 필요한 컬럼만 선택한다.
    
    # 필요 컬럼만 선택해서 record를 어디서 구성하는지에 대한 부분
    record = {
        # 식별자, 시각 정보
        "network_slice_id": normalized.get("network_slice_id", "0"),
        "timestamp": normalized.get("timestamp", ""),
        "device_id": normalized.get("device_id", "0"),

        # 트래픽/자원/품질 관련 핵심 수치
        "traffic_load_bps": coerce_float(normalized.get("traffic_load_bps", "0")),
        "traffic_type": coerce_int(normalized.get("traffic_type", "0")),
        "network_utilization_percent": coerce_float(normalized.get("network_utilization_percent", "0")),
        "latency_ms": coerce_float(normalized.get("latency_ms", "0")),
        "packet_loss_rate_percent": coerce_float(normalized.get("packet_loss_rate_percent", "0")),
        "signal_strength_dbm": coerce_float(normalized.get("signal_strength_dbm", "0")),
        "bandwidth_utilization_percent": coerce_float(normalized.get("bandwidth_utilization_percent", "0")),

        # 장애/과부하/처리량 지표
        "network_slice_failure": coerce_int(normalized.get("network_slice_failure", "0")),
        "qos_metric_throughput": coerce_float(normalized.get("qos_metric_throughput", "0")),
        "overload_status": coerce_int(normalized.get("overload_status", "0")),

        # 상황을 설명하는 범주형 코드
        "device_type": coerce_int(normalized.get("device_type", "0")),
        "region": coerce_int(normalized.get("region", "0")),
        "network_failure_count": coerce_int(normalized.get("network_failure_count", "0")),
        "time_of_day": coerce_int(normalized.get("time_of_day", "0")),
        "weather_conditions": coerce_int(normalized.get("weather_conditions", "0")),
    }
    
    # record를 바탕으로 LLM 학습용 단일 샘플 구조를 완성한다.
    return {
        "id": f"network-slicing-{index:05d}",
        "domain": "network_slicing_qos",

        # 모델에 보여줄 문제 문장
        "instruction": build_instruction(record),
        "input": "",

        # 모델이 ideally 생성해야 하는 정답 예시
        "output": build_expected_output(record),

        # 원본 row에서 추린 구조화 메타데이터
        "metadata": record,
    }


# [Flow 4] CSV 전체를 순회하며 JSONL split 파일(train/val/test)로 기록한다.
#
# 동작 방식:
# - inspect 단계에서 생성한 normalized_columns를 입력으로 받음
# - 각 row마다 build_record()를 호출해 payload 생성
# - choose_split() 결과에 따라 해당 split 파일에 한 줄 JSON 저장
#
# 자원 관리:
# - split별 파일 핸들을 미리 열고 finally에서 닫아 누수 방지
#
# 주의:
# - 현재는 모든 split 파일을 "w" 모드로 열기 때문에 기존 결과를 덮어쓴다.
# - append가 아니라 full regenerate 정책이라는 점을 주석으로 명시하는 것이 좋다.
def convert_csv_to_jsonl(path: Path, normalized_columns: dict[str, str]) -> dict[str, int]:
    split_paths = {name: PREP_ROOT / f"{name}.jsonl" for name in SPLIT_RULE}
    split_counts = {name: 0 for name in SPLIT_RULE}
    handles = {name: split_paths[name].open("w", encoding="utf-8") for name in SPLIT_RULE}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            # 실제 변환 단계:
            # CSV row를 한 줄씩 읽고 build_record에서 정규화/선택/구성을 수행한다.
            for index, row in enumerate(reader):
                # deterministic 규칙으로 어느 split에 넣을지 결정한다.
                split_name = choose_split(index)
                # 여기서 row -> payload 변환이 일어난다.
                payload = build_record(index, row, normalized_columns)
                # JSONL은 한 줄에 하나의 JSON object를 쓰는 형식이며,
                # split_name에 따라 train.jsonl / val.jsonl / test.jsonl 중 하나에 기록된다.
                handles[split_name].write(json.dumps(payload, ensure_ascii=False) + "\n")
                split_counts[split_name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return split_counts


# main entry point.
#
# 실행 시 보장하는 것:
# - raw / prepared 디렉터리 생성
# - 입력 CSV 탐색 및 대표 파일 선택
# - schema inspection 수행
# - train/val/test JSONL 생성
# - manifest / schema / preview 산출물 기록
#
# 실패 시나리오:
# - CSV가 없으면 split 생성은 건너뛰고 prep-only 상태의 manifest만 남긴다.
# - 즉 "완전 실패"보다 "현재 상태를 설명하는 산출물 생성"을 우선하는 구조다.
def main() -> None:
    # main 시작점.
    # 사용자는 보통 이 파일을 직접 실행하고, 아래 단계가 순서대로 수행된다.
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PREP_ROOT.mkdir(parents=True, exist_ok=True)

    # 전체 흐름:
    # [Flow 1] raw CSV 탐색 -> collect_raw_csvs
    # [Flow 2] 대표 CSV 선택 -> detect_primary_csv
    # [Flow 3] 헤더 inspection 및 컬럼 정규화 규칙 생성 -> inspect_csv
    # [Flow 4] 각 row를 JSONL 샘플로 변환 -> convert_csv_to_jsonl
    # [Flow 4-1] row normalize 후 record 구성 -> build_record
    # [Flow 5] manifest/schema/preview 저장 -> prep_manifest, csv_schema.json, csv_preview.json
    
    # [Flow 1] raw CSV 목록 수집
    raw_files = collect_raw_csvs()
    # [Flow 2] 대표 CSV 선택
    primary_csv = detect_primary_csv(raw_files)
    # [Flow 3] inspect_csv에서 헤더를 읽고 정규화 컬럼 맵을 만든다.
    csv_inspection = inspect_csv(primary_csv) if primary_csv is not None else None
    split_counts = None

    if primary_csv is not None and csv_inspection is not None:
        # inspect_csv에서 만든 normalized_columns를 실제 row 변환 단계에 넘긴다.
        # [Flow 4] convert_csv_to_jsonl에서 row를 한 줄씩 읽는다.
        # [Flow 4-1] build_record에서 row normalize -> 필요한 컬럼 선택 -> record 생성
        # [Flow 4-2] instruction/output/metadata를 포함한 payload를 JSONL로 저장
        # 여기서 split 규칙이 실제 파일 분할(train/val/test)에 적용된다.
        split_counts = convert_csv_to_jsonl(primary_csv, csv_inspection["normalized_columns"])

    # [Flow 5] 전처리 결과 요약 manifest를 구성한다.
    # splits 항목에는 각 split에 몇 개의 샘플이 기록되었는지가 들어가며,
    # split_rule 항목에는 왜 그런 분할 결과가 나왔는지 재현 가능한 규칙 자체를 남긴다.
    prep_manifest = {
        **DATASET_SPEC,
        "status": "ready" if split_counts is not None else DATASET_SPEC["status"],
        "raw_dir": str(RAW_ROOT),
        "prepared_dir": str(PREP_ROOT),
        "raw_csv_files": raw_files,
        "raw_csv_count": len(raw_files),
        "expected_primary_file": EXPECTED_RAW_FILE,
        "detected_primary_file": primary_csv.name if primary_csv is not None else None,
        "csv_inspection": csv_inspection,
        "splits": split_counts,
        "split_rule": SPLIT_RULE,
        "next_step": "Run test_dataset.sh against the generated train/val/test JSONL files if the server is ready.",
    }

    readme = """# 6G Network Slicing QoS dataset prep

- Source: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
- Status: prep-only
- Expected filename: 6G_network_slicing_qos_dataset_2345.csv
- Expected data type: CSV
- Core features: throughput, latency, packet loss rate
- Intended tasks: QoS prediction, slice optimization
- Intended LLM queries: latency-sensitive slice analysis, QoS anomaly detection, slice policy explanation
- Output files: train.jsonl, val.jsonl, test.jsonl

## What to place here

- Download the Kaggle dataset manually.
- Put `6G_network_slicing_qos_dataset_2345.csv` in this directory.
- If the exported filename differs, the prep script will inspect the first CSV it finds.

## Conversion rule

- One CSV row becomes one QoS analysis sample.
- The script writes `train.jsonl`, `val.jsonl`, and `test.jsonl` to the prepared directory.
- Split rule is deterministic `8:1:1` by row index.

## Planned next step

- Inspect source columns from `manifest.prep.json`.
- Start the model server.
- Run `test_dataset.sh` on the generated split files.
"""

    write_json(PREP_ROOT / "manifest.prep.json", prep_manifest)
    write_text(RAW_ROOT / "README.md", readme)

    if csv_inspection is not None:
        # [Flow 5] inspection 결과를 별도 파일로 남겨 원본 컬럼과 정규화 컬럼을 추적할 수 있게 한다.
        write_json(PREP_ROOT / "csv_schema.json", {
            "raw_file": csv_inspection["raw_file"],
            "row_count": csv_inspection["row_count"],
            "columns": csv_inspection["columns"],
            "normalized_columns": csv_inspection["normalized_columns"],
        })
        write_json(PREP_ROOT / "csv_preview.json", {
            "raw_file": csv_inspection["raw_file"],
            "preview_rows": csv_inspection["preview_rows"],
        })

    print("[dataset-prep] prepared target dataset scaffold")
    print(f"[dataset-prep] raw_dir={RAW_ROOT}")
    print(f"[dataset-prep] prepared_dir={PREP_ROOT}")
    print(f"[dataset-prep] detected_raw_csv={len(raw_files)}")
    if primary_csv is not None:
        print(f"[dataset-prep] primary_csv={primary_csv.name}")
        print(f"[dataset-prep] parsed_rows={csv_inspection['row_count']}")
    if split_counts is not None:
        print(f"[dataset-prep] split_counts={split_counts}")


if __name__ == "__main__":
    main()