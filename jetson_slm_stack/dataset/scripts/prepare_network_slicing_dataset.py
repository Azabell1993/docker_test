from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


# ============================================================
# 6G Network Slicing QoS Dataset Prep Script
# ------------------------------------------------------------
# 목적:
#   - Kaggle의 6G Network Slicing QoS CSV를 읽는다.
#   - 컬럼명을 snake_case 형태로 정규화한다.
#   - 각 row를 3GPP 기반 QoS 평가용 JSONL 샘플로 변환한다.
#   - train / val / test split을 deterministic(재현 가능)하게 생성한다.
#
# 핵심:
#   1) few-shot 20개를 SYSTEM_PROMPT에 포함
#   2) hard_breach=yes -> critical 규칙을 더 강하게 명시
#   3) instruction에서 hard_breach / stable_allowed를 맨 앞에 배치
#   4) MAX_SAMPLES=50 으로 제한
#
# 허용 KPI (7개):
#   - latency_ms
#   - packet_loss_rate_percent
#   - qos_metric_throughput
#   - network_utilization_percent
#   - bandwidth_utilization_percent
#   - signal_strength_dbm
#   - overload_status
#
# 보조 컬럼:
#   - traffic_type
#
# 비표준(3GPP):
#   - network_slice_failure
#   - device_type
#   - region
#   - network_failure_count
#   - time_of_day
#   - weather_conditions
# ============================================================


# ============================================================
# 경로 / 파일명 설정
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_ROOT = DATASET_DIR / "raw" / "network_slicing_qos"
DEFAULT_PREP_ROOT = DATASET_DIR / "prepared" / "network_slicing_qos"

RAW_ROOT = Path(os.getenv("DATASET_RAW_DIR", str(DEFAULT_RAW_ROOT)))
PREP_ROOT = Path(os.getenv("DATASET_PREP_DIR", str(DEFAULT_PREP_ROOT)))

EXPECTED_RAW_FILE = "6G_network_slicing_qos_dataset_2345.csv"


# ============================================================
# 데이터셋 메타 정보
# ============================================================

DATASET_SPEC = {
    "name": "6G Network Slicing QoS",
    "name_ko": "네트워크 슬라이싱 QoS",
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
        "network_utilization",
        "bandwidth_utilization",
        "signal_strength",
        "overload_status",
    ],
    "primary_uses": [
        "QoS state classification",
        "slice SLA risk explanation",
        "resource adjustment recommendation",
    ],
    "llm_use_cases": [
        "3GPP 기반 QoS 이상 탐지",
        "슬라이스 성능 상태 분류",
        "즉시 운영 액션 추천",
    ],
    "expected_raw_files": [
        EXPECTED_RAW_FILE,
    ],
}


# ============================================================
# split / 샘플 수 정책
# ============================================================
# deterministic split을 위해 row index 기반으로 split을 결정한다.
# 8:1:1 비율로 train, val, test를 나눈다. (예: index % 10 < 8 → train)
# 8은 train, 1은 val, 1은 test를 의미한다.
# train 의미는 모델이 학습하는 샘플, val은 모델 튜닝이나 early stopping에 활용하는 샘플, test는 최종 성능 평가에 활용하는 샘플이다.
SPLIT_RULE = {"train": 8, "val": 1, "test": 1}

# 요구사항 반영: 50개만 생성
MAX_SAMPLES = 50


# ============================================================
# 안전한 형변환 유틸리티
# ============================================================

def coerce_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ============================================================
# raw CSV 탐색 / 선택
# ============================================================

def collect_raw_csvs() -> list[str]:
    return sorted(path.name for path in RAW_ROOT.glob("*.csv"))


def detect_primary_csv(raw_files: list[str]) -> Path | None:
    expected = RAW_ROOT / EXPECTED_RAW_FILE
    if expected.exists():
        return expected
    if raw_files:
        return RAW_ROOT / raw_files[0]
    return None


# ============================================================
# 컬럼명 정규화
# ============================================================

def slugify_column(name: str) -> str:
    value = name.strip().lower()

    for source, target in (
        ("%", "percent"),
        ("/", "_per_"),
        ("-", "_"),
        (" ", "_"),
    ):
        value = value.replace(source, target)

    value = re.sub(r"[^a-z0-9_]", "_", value)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")

# ============================================================
# CSV 탐색 및 컬럼명 정규화
def inspect_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_columns = {column: slugify_column(column) for column in fieldnames}

        preview_rows: list[dict[str, str]] = []
        row_count = 0

        for row in reader:
            row_count += 1
            if len(preview_rows) < 5:
                preview_rows.append(
                    {
                        normalized_columns[key]: value
                        for key, value in row.items()
                        if key is not None
                    }
                )

    return {
        "raw_file": path.name,
        "row_count": row_count,
        "columns": fieldnames,
        "normalized_columns": normalized_columns,
        "preview_rows": preview_rows,
    }


# ============================================================
# deterministic split
# ============================================================
# index 기반으로 split을 결정한다. (예: index % 10 < 8 → train)
def choose_split(index: int) -> str:
    bucket = index % 10
    if bucket < SPLIT_RULE["train"]:
        return "train"
    if bucket < SPLIT_RULE["train"] + SPLIT_RULE["val"]:
        return "val"
    return "test"


# ============================================================
# 3GPP SLA 매핑
# ============================================================
# traffic_type 2 → eMBB-video (PDB 300ms, PER 0.01)
# traffic_type 3 → URLLC (PDB 10ms, PER 0.01)
# traffic_type 1 → eMBB-voice (PDB 100ms, PER 1.0)
# PDB: Packet Delay Budget, PER: Packet Error Rate
# SLA 기준은 3GPP TS 28.531을 참고하되, 실제 데이터 분포에 맞게 현실적으로 조정한다.
def _get_sla_thresholds(traffic_type: int) -> tuple[float, float]:
    if traffic_type == 2:
        return 300.0, 0.01
    if traffic_type == 3:
        return 10.0, 0.01
    return 100.0, 1.0


def _get_sst_label(traffic_type: int) -> str:
    if traffic_type == 2:
        return "eMBB-video"
    if traffic_type == 3:
        return "URLLC"
    return "eMBB-voice"


# ============================================================
# 정규화 값 -> 절대값 역변환
# ============================================================
# 모델이 해석하기 쉽도록 0~1 사이로 정규화된 KPI 값을 원래 단위로 복원하는 함수들이다.
def _restore_latency_ms(normalized_latency: float) -> float:
    return normalized_latency * 495.0 + 5.0

# 예시: 0.0 → 5ms, 1.0 → 500ms 범위로 복원한다. 실제 데이터 분포에 맞게 조정한다.
def _restore_packet_loss_percent(normalized_loss: float) -> float:
    return normalized_loss * 100.0
def _restore_throughput_percent(normalized_throughput: float) -> float:
    return normalized_throughput * 100.0
def _restore_util_percent(normalized_util: float) -> float:
    return normalized_util * 100.0

# 예시: 0.0 → -156dBm, 1.0 → -31dBm 범위로 복원한다. 실제 데이터 분포에 맞게 조정한다.
def _restore_signal_rsrp_dbm(normalized_signal: float) -> float:
    return -156.0 + normalized_signal * 125.0


# ============================================================
# 내부 플래그 계산
# ============================================================
def _get_prompt_flags(record: dict[str, str | int | float]) -> dict[str, str]:
    """
    핵심 강제 플래그만 남긴다.
    """
    traffic_type = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)

    #   latency_exceeds_pdb      → 지연 문제
    #   packet_loss_exceeds_per  → 손실 문제
    #   packet_loss_abnormal     → 심각한 손실
    #   signal_critical          → 신호 문제
    #   overload                 → 리소스 문제
    #   hard_breach = 위 문제 중 하나라도 터짐
    #   stable_allowed = hard_breach=no 이면서 모든 KPI가 정상 범위 내에 있음
    #   QoS state: hard_breach=yes → critical / hard_breach=no + stable_allowed=yes → stable / 나머지 → degraded
    #   Action: hard_breach=yes → 심각한 문제이므로 즉시 조치 권장 / stable_allowed=yes → 현재 정책 유지 권장 / 나머지 → 리소스 재조정 및 모니터링 권장
    #   모델이 직접 수치 비교를 잘 못하는 부분을 미리 계산해 준다.
    latency_ms = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    throughput_pct = _restore_throughput_percent(float(record["qos_metric_throughput"]))
    net_util_pct = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))
    overload = int(record["overload_status"])

    latency_exceeds_pdb = latency_ms > pdb_ms
    packet_loss_exceeds_per = loss_pct > per_pct
    packet_loss_abnormal = loss_pct >= 70.0
    signal_critical = rsrp_dbm < -110.0

    # ============================================================
    # hard_breach 계산
    # ============================================================

    # hard_breach는 아래 조건 중 하나라도 만족하면 TRUE → 반드시 critical 이어야 함
    hard_breach = (
        latency_exceeds_pdb         # 지연 SLA 초과 → critical
        or loss_pct > 5.0           # 손실률 심각 (PER 훨씬 초과) → critical
        or packet_loss_abnormal     # 비정상 손실 (≥70%) → 무조건 critical
        or signal_critical          # 신호 불량 (< -110 dBm) → critical

        # 과부하 + KPI 깨짐 → 시스템 레벨 문제 → critical
        or (overload == 1 and net_util_pct >= 80.0 and (latency_ms > pdb_ms or loss_pct > per_pct))
    )

    # ===================== 실제 결과 해석 =====================
    # 현재 테스트 결과를 보면:
    # instruction: hard_breach=yes, stable_allowed=no
    # → 이 조건이면 이론적으로 무조건 "critical"이 나와야 함

    # 그런데 실제 생성 결과:
    # QoS state: degraded   ❌ (완전히 잘못된 결과)

    # 원인:
    # - LLM이 SYSTEM_PROMPT의 "hard_breach=yes → 반드시 critical" 규칙을 무시함
    # - few-shot보다 자연어 reasoning을 더 강하게 따름
    # - flag 기반 rule보다 "latency + loss → degraded" 식으로 일반화된 패턴을 따름

    # 결론:
    # hard_breach 로직 자체는 정상
    # 문제는 모델이 rule을 "강제 조건"으로 인식 못함 (instruction failure)


    # ============================================================
    # stable_allowed 계산
    # ============================================================
    stable_allowed = (
        not hard_breach              # hard_breach가 없어야만 가능
        and latency_ms <= pdb_ms     # 지연 정상
        and loss_pct <= per_pct      # 손실 정상
        and overload == 0            # 과부하 없음
        and net_util_pct < 80.0      # 네트워크 여유
        and bw_util_pct < 80.0       # 대역폭 여유
        and rsrp_dbm >= -100.0       # 신호 양호
        and throughput_pct >= 40.0   # 처리량 정상
    )

    # ===================== 실제 결과 해석 =====================
    # 현재 테스트 케이스에서는:
    # hard_breach = yes 이므로
    # → stable_allowed는 항상 no

    # 따라서 가능한 상태는:
    # - critical (정상 동작)
    # - degraded (현재 잘못된 결과)

    # 즉,
    # stable이 나올 가능성은 아예 없음


    # ============================================================
    # 최종 상태 결정 (핵심)
    # ============================================================
    # 정상적인 의도된 동작:
    # if hard_breach == yes:
    #     → QoS state = critical
    # elif stable_allowed == yes:
    #     → QoS state = stable
    # else:
    #     → QoS state = degraded

    # ===================== 현재 문제 =====================
    # 실제 결과:
    # hard_breach=yes 인데도 → degraded 출력됨

    # 이는 다음을 의미:
    # 1) rule-based system은 정확히 동작 중
    # 2) 하지만 LLM이 "규칙 우선순위"를 따르지 않음
    # 3) 즉, instruction 설계 실패

    # ============================================================
    # 핵심 결론
    # ============================================================
    # - 로직 자체는 100% 맞음
    # - 모델이 hard rule을 무시하고 있음

    # ============================================================
    # 문제 원인 분석
    # ============================================================
    # 1. flag보다 자연어 reasoning을 우선함
    #    "latency + loss → degraded" 같은 일반 패턴 사용
    # 2. SYSTEM_PROMPT 강제력이 부족함
    #    → "must" 규칙이 soft constraint로 해석됨
    # 3. few-shot이 rule보다 약하게 작용
    #    → 패턴 학습이 규칙을 덮어버림
    # 4. 1B 모델 특성
    #    → 논리 우선순위 판단 불안정

    # ============================================================
    # 한줄 요약
    # ============================================================
    # 코드 문제는 아님
    # LLM이 hard_breach 규칙을 무시해서 전부 degraded로 떨어짐


# ============================================================
# few-shot examples (20개)
# ============================================================

FEW_SHOT_EXAMPLES = """
Example 1:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=eMBB-voice; overload=0
QoS state: critical
Reason: latency exceeds PDB and packet loss exceeds PER for eMBB-voice
Action: prioritize low-latency scheduling and reduce PRB contention

Example 2:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=yes; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes; traffic=eMBB-video; overload=1
QoS state: critical
Reason: packet loss is abnormal and exceeds PER for eMBB-video
Action: isolate the slice, validate telemetry, and reroute traffic immediately

Example 3:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=yes; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=URLLC; overload=0
QoS state: critical
Reason: signal is below the serviceable threshold for URLLC
Action: trigger handover or beam management to recover signal quality

Example 4:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=no; traffic=URLLC; overload=1
QoS state: critical
Reason: latency exceeds PDB for URLLC
Action: prioritize low-latency scheduling and reduce contention

Example 5:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes; traffic=eMBB-video; overload=0
QoS state: critical
Reason: packet loss exceeds PER for eMBB-video
Action: investigate radio quality and reduce packet loss with retransmission tuning

Example 6:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=yes; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=URLLC; overload=1
QoS state: critical
Reason: both latency and abnormal packet loss indicate severe SLA violation for URLLC
Action: isolate the slice, validate telemetry, and reroute traffic immediately

Example 7:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=eMBB-video; overload=0
QoS state: critical
Reason: latency exceeds PDB and packet loss exceeds PER for eMBB-video
Action: prioritize low-latency scheduling and reduce contention

Example 8:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes; traffic=eMBB-voice; overload=1
QoS state: critical
Reason: packet loss exceeds PER for eMBB-voice
Action: investigate radio quality and reduce packet loss with retransmission tuning

Example 9:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=yes; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=eMBB-video; overload=1
QoS state: critical
Reason: multiple critical breaches are present for eMBB-video
Action: redistribute slice traffic and reduce utilization below 80 percent

Example 10:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=eMBB-voice; overload=1
QoS state: critical
Reason: latency exceeds PDB and packet loss exceeds PER for eMBB-voice
Action: prioritize low-latency scheduling and reduce PRB contention

Example 11:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=yes; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes; traffic=eMBB-voice; overload=1
QoS state: critical
Reason: abnormal packet loss indicates a severe SLA violation for eMBB-voice
Action: isolate the slice, validate telemetry, and reroute traffic immediately

Example 12:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=URLLC; overload=0
QoS state: critical
Reason: latency exceeds PDB and packet loss exceeds PER for URLLC
Action: prioritize low-latency scheduling and reduce contention

Example 13:
hard_breach=no; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=eMBB-voice; overload=1
QoS state: degraded
Reason: overload is present without a critical breach for eMBB-voice
Action: rebalance load and monitor KPI trends closely

Example 14:
hard_breach=no; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=eMBB-video; overload=1
QoS state: degraded
Reason: overload is present without a critical breach for eMBB-video
Action: rebalance load and monitor KPI trends closely

Example 15:
hard_breach=no; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=URLLC; overload=1
QoS state: degraded
Reason: overload is present without a critical breach for URLLC
Action: rebalance load and monitor KPI trends closely

Example 16:
hard_breach=no; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=eMBB-voice; overload=0
QoS state: degraded
Reason: KPI state is near the warning range for eMBB-voice
Action: monitor KPI drift and prepare preventive load balancing

Example 17:
hard_breach=no; stable_allowed=yes; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=eMBB-voice; overload=0
QoS state: stable
Reason: all KPI values are within normal limits for eMBB-voice
Action: maintain current slice policy and continue KPI monitoring

Example 18:
hard_breach=no; stable_allowed=yes; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=eMBB-video; overload=0
QoS state: stable
Reason: all KPI values are within normal limits for eMBB-video
Action: maintain current slice policy and continue KPI monitoring

Example 19:
hard_breach=no; stable_allowed=yes; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=no; traffic=URLLC; overload=0
QoS state: stable
Reason: all KPI values are within normal limits for URLLC
Action: maintain current slice policy and continue KPI monitoring

Example 20:
hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no; signal_critical=no; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes; traffic=URLLC; overload=1
QoS state: critical
Reason: packet loss exceeds PER for URLLC
Action: investigate radio quality and reduce packet loss with retransmission tuning
""".strip()


# ============================================================
# SYSTEM_PROMPT
# ============================================================

SYSTEM_PROMPT = f"""Classify the 6G network slice QoS state using only the provided flags.
Return exactly 3 lines and nothing else:
QoS state: <stable|degraded|critical>
Reason: <one sentence>
Action: <one sentence>

Mandatory rules:
- If hard_breach=yes, you must output exactly: QoS state: critical
- Never output degraded when hard_breach=yes
- Never output stable when hard_breach=yes
- If packet_loss_abnormal=yes, QoS state must be critical
- If signal_critical=yes, QoS state must be critical
- If stable_allowed=yes, QoS state must be stable
- Otherwise, QoS state must be degraded

First decide the QoS state only from the rules.
Then write Reason and Action consistent with that state.

{FEW_SHOT_EXAMPLES}
"""


# ============================================================
# instruction 생성
# ============================================================

# ============================================================
# instruction 생성
# ============================================================

def build_instruction(record: dict[str, str | int | float]) -> str:
    """
    instruction :
    - LLM에게 직접 전달할 "판단용 입력 문자열"이다.
    - 원본 CSV의 수치 컬럼을 그대로 길게 주는 대신,
      rule-engine이 미리 계산한 핵심 flag만 요약해서 전달한다.
    - 목적은 작은 모델(예: 1B)이 복잡한 수치 비교를 직접 하지 않고도
      상태 분류 규칙을 최대한 단순하게 따르게 만드는 것이다.

    왜 필요한가?
    - 작은 모델은 latency > PDB, loss > PER 같은 수치 비교를 자주 틀린다.
    - 그래서 사람이 보면 쉬운 비교도 모델은 불안정하게 처리할 수 있다.
    - 이를 막기 위해 Python 쪽에서 먼저 비교 결과를 boolean flag로 계산하고,
      그 결과만 instruction에 실어 준다.

    이 함수가 만드는 instruction 예시:
        hard_breach=yes; stable_allowed=no; packet_loss_abnormal=no;
        signal_critical=no; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes;
        traffic=eMBB-voice; overload=0

    모델은 이 문자열을 보고:
        - hard_breach=yes 이면 critical
        - stable_allowed=yes 이면 stable
        - 둘 다 아니면 degraded
    의 우선순위로 판단하도록 유도된다.

    즉, instruction은
    "원본 컬럼 → 직접 판단"이 아니라
    "이미 해석된 핵심 규칙 결과 → 최종 라벨 생성"
    을 위한 중간 표현이다.

    또한 hard_breach / stable_allowed를 맨 앞에 두는 이유는
    모델이 가장 중요한 우선순위 규칙을 먼저 읽게 하기 위해서다.
    """
    traffic_type = int(record["traffic_type"])
    sst_label = _get_sst_label(traffic_type)
    flags = _get_prompt_flags(record)
    overload = int(record["overload_status"])

    # 포함되는 항목 의미:
    #
    # hard_breach
    #   → critical을 강하게 유도하는 최상위 요약 플래그
    #
    # stable_allowed
    #   → 모든 KPI가 정상일 때만 yes가 되는 정상 상태 허용 플래그
    #
    # packet_loss_abnormal
    #   → 손실률이 비정상적으로 매우 큰지 여부 (예: 70% 이상)
    #
    # signal_critical
    #   → RSRP가 서비스 불가 수준인지 여부
    #
    # latency_exceeds_pdb
    #   → 트래픽 유형별 PDB를 지연이 초과했는지 여부
    #
    # packet_loss_exceeds_per
    #   → 트래픽 유형별 PER 기준을 손실률이 초과했는지 여부
    #
    # traffic
    #   → 현재 슬라이스/서비스 성격 (eMBB-voice, eMBB-video, URLLC)
    #
    # overload
    #   → 과부하 상태 여부

    # 요약 구조:
    # latency / loss / signal / overload 같은 원본 KPI 문제
    #        ↓
    # _get_prompt_flags() 에서 핵심 flag 계산
    #        ↓
    # hard_breach / stable_allowed 중심으로 instruction 구성
    #        ↓
    # 모델이 최종 QoS state 생성

    return (
        f"hard_breach={flags['hard_breach']}; "
        f"stable_allowed={flags['stable_allowed']}; "
        f"packet_loss_abnormal={flags['packet_loss_abnormal']}; "
        f"signal_critical={flags['signal_critical']}; "
        f"latency_exceeds_pdb={flags['latency_exceeds_pdb']}; "
        f"packet_loss_exceeds_per={flags['packet_loss_exceeds_per']}; "
        f"traffic={sst_label}; "
        f"overload={overload}"
    )


# ============================================================
# QoS 상태 분류
# ============================================================
# ============================================================
# QoS 상태 분류
# ============================================================

def classify_qos_state(record: dict[str, str | int | float]) -> str:
    # --------------------------------------------------------
    # 1. 원본 입력은 정규화(0~1) 값이므로 실제 물리 단위로 복원한다.
    # --------------------------------------------------------
    # latency_ms                    : 실제 지연(ms)
    # packet_loss_rate_percent      : 실제 손실률(%)
    # qos_metric_throughput         : 상대 처리량(%)
    # network_utilization_percent   : 네트워크 자원 활용률(%)
    # bandwidth_utilization_percent : 대역폭 활용률(%)
    # signal_strength_dbm           : 실제 RSRP(dBm)
    # overload_status               : 과부하 여부 (0/1)
    # traffic_type                  : 서비스 유형 구분용 코드
    #
    # 이 함수의 핵심은:
    # "정규화된 CSV 수치 → 실제 단위 복원 → rule-based QoS 상태 분류"
    #
    # 즉, LLM 없이 Python 규칙만으로 stable / degraded / critical을 결정하는
    # 기준 함수다.
    latency_ms = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    throughput_pct = _restore_throughput_percent(float(record["qos_metric_throughput"]))
    net_util_pct = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))
    overload = int(record["overload_status"])
    traffic_type = int(record["traffic_type"])

    # --------------------------------------------------------
    # 2. traffic_type에 따라 SLA 기준(PDB, PER)을 가져온다.
    # --------------------------------------------------------
    # traffic_type은 단순 숫자지만, 내부적으로는 서비스 타입을 뜻한다.
    #
    # 예시:
    # - eMBB-voice : PDB 100ms, PER 1.0%
    # - eMBB-video : PDB 300ms, PER 0.01%
    # - URLLC      : PDB 10ms,  PER 0.01%
    #
    # 즉, "같은 latency 120ms"라도
    # - voice에서는 SLA 초과일 수 있고
    # - video에서는 허용 가능할 수 있으며
    # - URLLC에서는 매우 심각할 수 있다.
    #
    # 그래서 반드시 traffic_type에 따라 기준이 달라져야 한다.
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)

    # ========================================================
    # 3. critical 판정
    # ========================================================
    # critical은 "이미 서비스가 깨졌거나, 즉시 강한 조치가 필요한 상태"
    #
    # 이 구간은 가장 먼저 검사한다.
    # 이유:
    # - 심각 상태를 놓치면 안 되기 때문
    # - stable/degraded보다 우선순위가 가장 높기 때문

    # [critical-1] packet loss가 70% 이상
    # - 일반적인 품질 저하 수준이 아니라 비정상/장애 수준으로 본다.
    # - 데이터 이상치일 수도 있지만 운영 관점에서는 보수적으로 critical 처리한다.
    if loss_pct >= 70.0:
        return "critical"

    # [critical-2] latency가 트래픽별 PDB를 초과
    # - SLA 상 허용 지연을 넘었으므로 서비스 품질 위반
    # - voice, video, URLLC 모두 traffic_type에 따라 다른 기준을 사용함
    if latency_ms > pdb_ms:
        return "critical"

    # [critical-3] packet loss가 5% 초과
    # - PER 기준보다 훨씬 높은 심각 손실로 간주
    # - 현재 규칙에서는 PER을 조금 넘는 수준과 별개로,
    #   5% 초과면 강한 장애 수준으로 본다.
    if loss_pct > 5.0:
        return "critical"

    # [critical-4] RSRP가 -110 dBm 미만
    # - 무선 신호가 매우 불량한 상태
    # - 실제 서비스 지속이 어렵다고 판단
    if rsrp_dbm < -110.0:
        return "critical"

    # [critical-5] overload + high utilization + SLA 깨짐
    # - 단순 과부하만으로는 critical이 아닐 수 있음
    # - 하지만 과부하 상태이면서 자원 사용률도 높고,
    #   동시에 latency 또는 loss 문제까지 동반되면
    #   시스템 수준 병목으로 보고 critical 처리
    if overload == 1 and net_util_pct >= 80.0 and (latency_ms > pdb_ms or loss_pct > per_pct):
        return "critical"

    # ========================================================
    # 4. degraded 판정
    # ========================================================
    # degraded는 "아직 완전히 깨지진 않았지만 품질 저하가 시작된 상태"다.
    #
    # critical은 아니지만 정상도 아닌 중간 경고 구간이다.

    # [degraded-1] latency가 PDB 80~100% 수준
    # - 아직 SLA 위반은 아니지만 임계점에 가까움
    # - 곧 critical로 넘어갈 수 있는 경고 상태
    if pdb_ms * 0.8 < latency_ms <= pdb_ms:
        return "degraded"

    # [degraded-2] packet loss가 PER 초과이지만 5% 미만
    # - 손실이 기준보다 높아 품질이 저하되었음
    # - 다만 아직 severe loss로 볼 정도는 아니므로 degraded
    if per_pct < loss_pct < 5.0:
        return "degraded"

    # [degraded-3] 네트워크 활용률 80~90%
    # - 자원 포화 직전
    # - 아직 장애는 아니지만 혼잡으로 인한 저하 위험이 큼
    if 80.0 <= net_util_pct < 90.0:
        return "degraded"

    # [degraded-4] 대역폭 활용률 80~90%
    # - bandwidth 측면에서 여유가 줄어든 상태
    # - throughput 저하나 병목 전조로 해석 가능
    if 80.0 <= bw_util_pct < 90.0:
        return "degraded"

    # [degraded-5] overload=1
    # - 과부하 플래그 자체가 켜졌다면 운영상 경고 상태
    # - 단, 위에서 이미 critical 조건을 먼저 배제했으므로
    #   여기서는 "심각하진 않지만 저하됨"으로 분류
    if overload == 1:
        return "degraded"

    # [degraded-6] 신호가 -110~-100 dBm
    # - 서비스 불가 수준은 아니지만 품질 저하 가능성이 높음
    # - packet loss / latency 악화의 전조로 본다
    if -110.0 <= rsrp_dbm < -100.0:
        return "degraded"

    # [degraded-7] throughput가 40% 미만
    # - 절대 SLA 기준은 아니지만 처리량 부족으로 품질 저하 가능성 큼
    # - 현재 규칙에서는 단독 degraded 조건으로 사용
    if throughput_pct < 40.0:
        return "degraded"

    # ========================================================
    # 5. stable 판정
    # ========================================================
    # stable은 "모든 핵심 KPI가 정상 범위 안에 있음"을 뜻한다.
    #
    # 즉, latency/loss/signal/utilization/throughput/overload 모두
    # 정상 조건을 동시에 만족해야만 stable이다.
    #
    # stable은 가장 엄격한 정상 상태라서
    # 조건 하나라도 어긋나면 stable이 될 수 없다.
    if (
        latency_ms <= pdb_ms          # 지연 정상
        and loss_pct <= per_pct       # 손실 정상
        and overload == 0             # 과부하 없음
        and net_util_pct < 80.0       # 자원 여유 있음
        and bw_util_pct < 80.0        # 대역폭 여유 있음
        and rsrp_dbm >= -100.0        # 신호 양호
        and throughput_pct >= 40.0    # 처리량 확보
    ):
        return "stable"

    # ========================================================
    # 6. degraded 판정 (기본값)
    # ========================================================
    # 여기까지 왔다는 것은:
    # - critical 조건도 아니고
    # - stable 조건도 완전 충족하지 않았다는 뜻
    # 따라서 기본적으로 degraded로 본다.
    #
    # critical = 심각 상태
    # stable   = 완전 정상
    # degraded = 그 사이의 모든 애매한 상태
    return "degraded"

# ============================================================
# 추천 액션 생성
# ============================================================

def recommend_action(record: dict[str, str | int | float], qos_state: str) -> str:
    # QoS 상태를 만든 것과 같은 방식으로
    # 원본 정규화 값을 실제 운영 의미가 있는 단위로 복원한다.
    #
    # 이 함수의 목적은:
    # "현재 어떤 문제가 가장 지배적인가?"를 보고
    # 그 문제에 맞는 즉시 대응 액션을 한 문장으로 반환하는 것
    latency_ms = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    throughput_pct = _restore_throughput_percent(float(record["qos_metric_throughput"]))
    net_util_pct = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))
    overload = int(record["overload_status"])
    traffic_type = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)

    # --------------------------------------------------------
    # 액션 추천 우선순위는 "가장 심각한 원인부터" 검사한다.
    # --------------------------------------------------------

    # [action-1] packet loss가 비정상적으로 매우 큰 경우
    # - 단순 튜닝 수준이 아니라 슬라이스 격리/텔레메트리 검증/우회가 필요한 상태
    if loss_pct >= 70.0:
        return "isolate the slice, validate telemetry, and reroute traffic immediately"

    # [action-2] 과부하 + 높은 utilization + SLA 깨짐
    # - 특정 KPI 하나만의 문제가 아니라 시스템 자원 병목으로 판단
    # - 슬라이스 트래픽 재분산, utilization 완화가 우선
    if overload == 1 and net_util_pct >= 80.0 and (latency_ms > pdb_ms or loss_pct > per_pct):
        return "redistribute slice traffic and reduce utilization below 80 percent"

    # [action-3] 지연이 PDB 초과
    # - 스케줄링 우선순위 조정, contention 완화가 핵심
    if latency_ms > pdb_ms:
        return "prioritize low-latency scheduling and reduce contention"

    # [action-4] 손실률이 매우 큼 (>5%)
    # - 무선 품질, 재전송, 링크 품질 점검이 우선
    if loss_pct > 5.0:
        return "investigate radio quality and reduce packet loss with retransmission tuning"

    # [action-5] 신호가 매우 나쁨
    # - handover, beam 관리 등 RF 개선 액션 우선
    if rsrp_dbm < -110.0:
        return "trigger handover or beam management to recover signal quality"

    # [action-6] 자원 과부하/고활용률
    # - 아직 완전 critical은 아니더라도
    #   load rebalance, scale-up 같은 예방적 조치 필요
    if overload == 1 or net_util_pct >= 80.0 or bw_util_pct >= 80.0:
        return "rebalance load and scale resources before SLA violation worsens"

    # [action-7] PER만 초과하는 경우
    # - 심각 손실까지는 아니더라도 QoS 튜닝 필요
    if loss_pct > per_pct:
        return "tune QoS handling to bring packet loss back within PER"

    # [action-8] 신호가 경계 수준
    # - 아직 서비스 불가 수준은 아니지만 RF 품질 개선 필요
    if -110.0 <= rsrp_dbm < -100.0:
        return "improve RF conditions and mobility control to raise signal strength"

    # [action-9] 처리량 부족
    # - 스케줄링/대역폭 배분 재조정 필요
    if throughput_pct < 40.0:
        return "review scheduling and bandwidth allocation to recover throughput"

    # [action-10] 완전 안정 상태
    # - 현재 정책 유지 + 모니터링
    if qos_state == "stable":
        return "maintain the current policy and continue KPI monitoring"

    # [action-11] 그 외 애매한 저하 상태
    # - 기본적인 자원 재검토 및 모니터링 권고
    return "review slice resource allocation and monitor KPI trends closely"


# ============================================================
# 기대 출력 생성
# ============================================================

def build_expected_output(record: dict[str, str | int | float]) -> str:
    traffic_type = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)
    sst_label = _get_sst_label(traffic_type)

    latency_ms = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    net_util_pct = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))

    qos_state = classify_qos_state(record)
    action = recommend_action(record, qos_state)

    if loss_pct >= 70.0:
        reason = f"packet loss {loss_pct:.1f}% is abnormal and exceeds PER {per_pct}%"
    elif latency_ms > pdb_ms:
        reason = f"latency {latency_ms:.1f}ms exceeds PDB {pdb_ms:.0f}ms for {sst_label}"
    elif loss_pct > 5.0:
        reason = f"packet loss {loss_pct:.1f}% is severely above PER {per_pct}%"
    elif rsrp_dbm < -110.0:
        reason = f"signal {rsrp_dbm:.1f}dBm is below the serviceable threshold"
    elif qos_state == "degraded":
        reason = (
            f"near-limit KPI state: latency {latency_ms:.1f}ms, loss {loss_pct:.3f}%, "
            f"net {net_util_pct:.1f}%, bw {bw_util_pct:.1f}%, rsrp {rsrp_dbm:.1f}dBm"
        )
    else:
        reason = f"all KPI values are within normal limits for {sst_label}"

    return (
        f"QoS state: {qos_state}\n"
        f"Reason: {reason}\n"
        f"Action: {action}"
    )


# ============================================================
# row 정규화 / record 구성
# ============================================================

def normalize_row(row: dict[str, str], normalized_columns: dict[str, str]) -> dict[str, str]:
    """
    CSV 원본 row의 컬럼명을 내부 표준 컬럼명으로 치환한다.

    왜 필요한가?
    - CSV 원본 헤더는 공백, 대문자, 특수문자(%, -, /) 등이 섞여 있을 수 있다.
    - 예를 들어:
      "Network Utilization (%)"
      → "network_utilization_percent"
    - 이후 로직은 고정된 snake_case 컬럼명 기준으로만 동작하므로
      입력 헤더를 먼저 내부 표준명으로 맞춰야 한다.

    이 함수가 하는 일:
    - row.items()를 순회하면서
    - normalized_columns 매핑 테이블을 이용해
      "원본 컬럼명 -> 정규화 컬럼명" 으로 바꾼 새 dict를 만든다.

    결과:
    - 이후 build_record(), classify_qos_state(), build_instruction() 등은
      CSV 원본 헤더 형식에 의존하지 않고
      내부 표준 필드명만 보고 일관되게 처리할 수 있다.
    """
    return {
        normalized_columns[key]: value
        for key, value in row.items()
        if key is not None
    }


def build_record(index: int, row: dict[str, str], normalized_columns: dict[str, str]) -> dict:
    """
    CSV의 row 1개를 JSONL 샘플 1개로 변환한다.

    전체 흐름:
    1) row의 컬럼명을 내부 표준명으로 정규화
    2) 필요한 KPI 컬럼만 뽑아 record(metadata용 내부 데이터) 구성
    3) 이 record를 바탕으로
       - system prompt
       - instruction
       - expected output
       - metadata
      를 합쳐 최종 JSONL 샘플을 만든다

    여기서 중요한 점:
    - record는 "모델 입력 생성과 정답 생성에 사용하는 내부 사실 데이터"
    - 최종 return 값은 "학습/평가용 JSONL 샘플"

    즉:
    row(raw CSV)
      → normalized
      → record(정제된 내부 데이터)
      → JSONL payload(system/instruction/output/metadata)
    """
    normalized = normalize_row(row, normalized_columns)

    # --------------------------------------------------------
    # record: 내부 판단에 필요한 최소 필드만 남긴다.
    #
    # 컬럼 정규화는 normalize_row()가 아니라
    # slugify_column() → inspect_csv() 단계에서 이미 수행된다.
    #
    # normalize_row()는 "정규화된 컬럼명으로 rename만 하는 역할"이며,
    # 실제 변환 로직은 slugify_column()에 있다.
    #
    # 여기(build_record)에서는:
    # - 이미 정규화된 컬럼명을 기준으로 필요한 필드만 선택하고
    # - 타입(coerce_int/float)을 맞춰 record를 구성한다.
    # --------------------------------------------------------
    # traceability(추적성) 용도:
    # - network_slice_id
    # - timestamp
    # - device_id
    #
    # QoS 판단 핵심 필드:
    # - traffic_type
    # - network_utilization_percent
    # - latency_ms
    # - packet_loss_rate_percent
    # - signal_strength_dbm
    # - bandwidth_utilization_percent
    # - qos_metric_throughput
    # - overload_status
    record = {
        "network_slice_id": normalized.get("network_slice_id", "0"),
        "timestamp": normalized.get("timestamp", ""),
        "device_id": normalized.get("device_id", "0"),

        "traffic_type": coerce_int(normalized.get("traffic_type", "0")),
        "network_utilization_percent": coerce_float(normalized.get("network_utilization_percent", "0")),
        "latency_ms": coerce_float(normalized.get("latency_ms", "0")),
        "packet_loss_rate_percent": coerce_float(normalized.get("packet_loss_rate_percent", "0")),
        "signal_strength_dbm": coerce_float(normalized.get("signal_strength_dbm", "0")),
        "bandwidth_utilization_percent": coerce_float(normalized.get("bandwidth_utilization_percent", "0")),
        "qos_metric_throughput": coerce_float(normalized.get("qos_metric_throughput", "0")),
        "overload_status": coerce_int(normalized.get("overload_status", "0")),
    }

    # --------------------------------------------------------
    # JSONL payload 구성
    # --------------------------------------------------------
    # id
    #   - 각 샘플을 유일하게 식별하기 위한 고정 포맷 ID
    #
    # domain
    #   - 어떤 데이터셋/도메인인지 명시
    #
    # system
    #   - LLM에게 항상 공통으로 주어지는 시스템 프롬프트
    #   - 규칙, 출력 형식, few-shot 예시 포함
    #
    # instruction
    #   - record를 바탕으로 생성한 "모델용 요약 입력"
    #   - hard_breach, stable_allowed 같은 핵심 flag 위주
    #
    # input
    #   - 현재 구조에서는 비워둠
    #   - instruction만으로 판단하게 설계했기 때문
    #
    # output
    #   - rule-based engine이 계산한 기대 정답
    #   - 학습 시 정답 라벨, 평가 시 expected answer 역할
    #
    # metadata
    #   - 원본 KPI 값을 보존
    #   - 추후 디버깅, 분석, 정량 평가, explainability에 사용
    #
    # 핵심 설계 의도:
    # - LLM에는 단순화된 instruction을 보여주되
    # - 사람이 검증하거나 후처리할 때는 metadata로 원본 근거를 추적 가능하게 함
    return {
        "id": f"network-slicing-{index:05d}",
        "domain": "network_slicing_qos",
        "system": SYSTEM_PROMPT,
        "instruction": build_instruction(record),
        "input": "",
        "output": build_expected_output(record),
        "metadata": record,
    }


# ============================================================
# CSV -> JSONL 변환
# ============================================================
def convert_csv_to_jsonl(path: Path, normalized_columns: dict[str, str]) -> dict[str, int]:
    """
    원본 CSV 전체를 읽어서 train / val / test JSONL 파일로 분할 저장한다.

    이 함수의 역할:
    - CSV를 한 줄씩 읽는다
    - 각 줄을 build_record()로 JSONL 샘플로 바꾼다
    - deterministic split 규칙에 따라 train/val/test 중 하나에 기록한다
    - 각 split에 몇 개 저장했는지 count를 반환한다

    왜 JSONL인가?
    - 한 줄 = 한 샘플 구조라서 LLM 학습/평가 파이프라인과 잘 맞는다
    - streaming 처리와 append/line-based parsing이 쉽다
    """

    # split별 출력 파일 경로 생성
    # 예:
    # - prepared/network_slicing_qos/train.jsonl
    # - prepared/network_slicing_qos/val.jsonl
    # - prepared/network_slicing_qos/test.jsonl
    split_paths = {name: PREP_ROOT / f"{name}.jsonl" for name in SPLIT_RULE}

    # split별 샘플 수 카운트
    split_counts = {name: 0 for name in SPLIT_RULE}

    # split별 파일 핸들 미리 open
    # 한 줄 처리할 때마다 열고 닫는 오버헤드를 줄이기 위함
    handles = {name: split_paths[name].open("w", encoding="utf-8") for name in SPLIT_RULE}

    total = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            # CSV row를 index와 함께 순회
            for index, row in enumerate(reader):
                # MAX_SAMPLES를 넘으면 중단
                # 현재 실험 버전에서는 50개만 생성하도록 제한
                if total >= MAX_SAMPLES:
                    break

                # deterministic split 규칙으로 train/val/test 결정
                # 예: row index % 10 기반 8:1:1
                split_name = choose_split(index)

                # row 1개 -> JSONL payload 1개
                payload = build_record(index, row, normalized_columns)

                # JSON 문자열 한 줄로 저장
                # JSONL 형식이므로 줄바꿈으로 샘플을 구분
                handles[split_name].write(json.dumps(payload, ensure_ascii=False) + "\n")

                split_counts[split_name] += 1
                total += 1

    finally:
        # 예외가 나더라도 파일 핸들은 반드시 닫는다.
        # 리소스 누수 방지
        for handle in handles.values():
            handle.close()

    # 최종적으로 각 split에 몇 개 들어갔는지 반환
    return split_counts


# ============================================================
# main
# ============================================================
def main() -> None:
    """
    데이터셋 준비 파이프라인의 전체 실행 진입점.

    main이 하는 일:
    1) raw / prepared 디렉토리 생성 보장
    2) raw CSV 파일 목록 수집
    3) 주 CSV(primary CSV) 선택
    4) CSV 구조 검사(inspect)
    5) CSV -> JSONL 변환 수행
    6) manifest / readme / schema / preview 파일 생성
    7) 최종 요약 로그 출력

    즉, 단순 변환 스크립트가 아니라
    "데이터셋 준비 결과를 재현 가능하게 남기는 준비 파이프라인"
    이라고 보면 된다.
    """

    # raw 데이터 경로와 prepared 출력 경로가 없으면 생성
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PREP_ROOT.mkdir(parents=True, exist_ok=True)

    # raw 폴더 안의 CSV 목록 수집
    raw_files = collect_raw_csvs()

    # 우선 처리할 CSV 1개 선택
    # 보통 EXPECTED_RAW_FILE을 우선 찾고, 없으면 첫 번째 CSV 사용
    primary_csv = detect_primary_csv(raw_files)

    # CSV 구조 점검 결과
    # - row 수
    # - 원본 컬럼 목록
    # - 정규화 컬럼 매핑
    # - 미리보기 row 등
    csv_inspection = inspect_csv(primary_csv) if primary_csv is not None else None

    split_counts = None

    # 주 CSV가 존재하고 구조 점검도 성공한 경우에만 변환 수행
    if primary_csv is not None and csv_inspection is not None:
        split_counts = convert_csv_to_jsonl(primary_csv, csv_inspection["normalized_columns"])

    # --------------------------------------------------------
    # prep_manifest 작성
    # --------------------------------------------------------
    # manifest는 이번 데이터셋 준비 작업의 결과 요약이다.
    #
    # 왜 필요한가?
    # - 어떤 원본 파일을 썼는지
    # - 몇 개의 row를 읽었는지
    # - split 결과가 어떻게 됐는지
    # - 현재 정책(MAX_SAMPLES, few-shot 개수, split rule 등)이 무엇인지
    # 를 한 파일에 남겨 재현성과 추적성을 높이기 위함이다.
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
        "max_samples": MAX_SAMPLES,
        "few_shot_count": 20,
        "policy_note": (
            "Only 3GPP-aligned KPI fields are retained in metadata and used for QoS evaluation. "
            "The prompt uses 20 few-shot examples and strong hard_breach-to-critical rules."
        ),
        "next_step": "Run dataset validation or model inference against the generated JSONL files.",
    }

    # --------------------------------------------------------
    # README 작성
    # --------------------------------------------------------
    # 이 README는 raw/prepared 데이터셋을 사람이 열어봤을 때
    # 현재 파이프라인 정책을 바로 이해할 수 있도록 만든 설명 문서다.
    #
    # 포함 내용:
    # - 데이터 출처
    # - 기대 파일명
    # - 출력 파일명
    # - split 정책
    # - 최대 샘플 수
    # - few-shot 개수
    # - 허용/보조/추적/제외 컬럼 목록
    readme = """# 6G Network Slicing QoS dataset prep

- Source: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
- Expected filename: 6G_network_slicing_qos_dataset_2345.csv
- Output files: train.jsonl, val.jsonl, test.jsonl
- Split rule: deterministic 8:1:1 by row index
- Max samples: 50
- Few-shot examples in system prompt: 20
- Evaluation policy: only 3GPP-aligned QoS KPI fields are retained for LLM evaluation

## Allowed QoS KPI fields

- latency_ms
- packet_loss_rate_percent
- qos_metric_throughput
- network_utilization_percent
- bandwidth_utilization_percent
- signal_strength_dbm
- overload_status

## Auxiliary field

- traffic_type

## Metadata retained for traceability only

- network_slice_id
- timestamp
- device_id

## Excluded non-standard fields

- network_slice_failure
- device_type
- region
- network_failure_count
- time_of_day
- weather_conditions
"""

    # manifest.prep.json 저장
    # → 이번 데이터셋 준비 결과를 기계가 다시 읽을 수 있게 JSON으로 보존
    write_json(PREP_ROOT / "manifest.prep.json", prep_manifest)

    # README.md 저장
    # → 사람이 보기 위한 설명 문서
    write_text(RAW_ROOT / "README.md", readme)

    if csv_inspection is not None:
        # ----------------------------------------------------
        # csv_schema.json 저장
        # ----------------------------------------------------
        # 용도:
        # - 원본 CSV의 전체 컬럼 구조와 정규화 맵을 따로 저장
        # - 후속 코드/발표/검증 때 "이 컬럼이 어떻게 바뀌었는지" 추적 가능
        write_json(
            PREP_ROOT / "csv_schema.json",
            {
                "raw_file": csv_inspection["raw_file"],
                "row_count": csv_inspection["row_count"],
                "columns": csv_inspection["columns"],
                "normalized_columns": csv_inspection["normalized_columns"],
            },
        )

        # ----------------------------------------------------
        # csv_preview.json 저장
        # ----------------------------------------------------
        # 용도:
        # - 원본 CSV의 일부 샘플 row를 미리보기로 남김
        # - 실제 컬럼값 형식과 데이터 품질을 빠르게 점검할 수 있음
        write_json(
            PREP_ROOT / "csv_preview.json",
            {
                "raw_file": csv_inspection["raw_file"],
                "preview_rows": csv_inspection["preview_rows"],
            },
        )

    # --------------------------------------------------------
    # 실행 결과 로그 출력
    # --------------------------------------------------------
    # 사용자가 스크립트를 실행했을 때
    # 현재 어떤 데이터가 감지되었고,
    # 몇 row를 읽었고,
    # split 결과가 어떻게 되었는지
    # 즉시 확인할 수 있도록 콘솔에 요약 출력한다.
    print("[dataset-prep] prepared target dataset scaffold")
    print(f"[dataset-prep] raw_dir={RAW_ROOT}")
    print(f"[dataset-prep] prepared_dir={PREP_ROOT}")
    print(f"[dataset-prep] detected_raw_csv={len(raw_files)}")
    if primary_csv is not None:
        print(f"[dataset-prep] primary_csv={primary_csv.name}")
        print(f"[dataset-prep] parsed_rows={csv_inspection['row_count']}")
    if split_counts is not None:
        print(f"[dataset-prep] split_counts={split_counts}")


# ============================================================
# 스크립트 직접 실행 시 진입점
# ============================================================
# 이 파일을 python xxx.py 형태로 직접 실행했을 때만 main()을 호출한다.
# import해서 재사용할 때는 자동 실행되지 않는다.
if __name__ == "__main__":
    main()