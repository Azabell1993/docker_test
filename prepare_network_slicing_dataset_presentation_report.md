# prepare_network_slicing_dataset.py 레포트

> **대상 파일**: `jetson_slm_stack/dataset/scripts/prepare_network_slicing_dataset.py`  
> **작성 기준**: 2026-03-30 현재 코드 실체 + 평가 실행 결과 반영  
> **평가 환경**: Nvidia DGX · Llama 3.2 1B-Instruct · test split 30건

---

## 1. 스크립트 개요

| 항목 | 내용 |
|---|---|
| 파일 | `jetson_slm_stack/dataset/scripts/prepare_network_slicing_dataset.py` |
| 입력 | `6G_network_slicing_qos_dataset_2345.csv` (Kaggle, 2345 rows) |
| 출력 | `train.jsonl` / `val.jsonl` / `test.jsonl` / `manifest.prep.json` / `csv_schema.json` / `csv_preview.json` |
| 분할 규칙 | deterministic 8:1:1 (row index % 10 기준) |
| 최대 샘플 수 설정 | `MAX_SAMPLES = 50` (실제 실행 결과: train 400 / val 50 / test 50) |
| 최적화 대상 | Jetson Orin Nano + Llama 3.2 1B-Instruct (Nvidia DGX로 검증) |

CSV 한 row를 `(system, instruction, output)` 쌍으로 바꾸는 **규칙 기반 데이터셋 생성기**이며,  
모델이 학습·추론 시 볼 입력(`instruction`)과 정답(`output`)을 **동시에 이 스크립트가 만들어낸다**.

---

## 2. 전체 처리 흐름

```
[Flow 1] raw CSV 목록 수집          collect_raw_csvs()
[Flow 2] 대표 CSV 선택               detect_primary_csv()
[Flow 3] 헤더 inspection + 정규화 맵 inspect_csv() + slugify_column()
[Flow 4] row → JSONL 샘플 변환      convert_csv_to_jsonl() → build_record()
           ├─ row 정규화               normalize_row()
           ├─ record 구성              coerce_float / coerce_int + 허용 KPI만 선별
           ├─ instruction 생성         build_instruction()
           │    ├─ _get_sla_thresholds()  traffic_type → PDB / PER
           │    ├─ _restore_*()           정규화 값 → 물리 값
           │    └─ _get_prompt_flags()    물리 값 기준 boolean 플래그 6개
           └─ output 생성              build_expected_output()
                ├─ classify_qos_state()   물리 값 → stable/degraded/critical
                ├─ recommend_action()     QoS + KPI → 한 문장 액션
                └─ reason 문장 조합
[Flow 5] manifest / schema / preview / README 저장
```

---

## 3. 원본 컬럼 ↔ 정규화 키 매핑

`slugify_column()`이 원본 CSV 헤더를 snake_case key로 변환한다.  
변환 규칙: 소문자화 → `%→percent`, `/→_per_`, `-·공백→_` → 비허용 문자 `_` 치환 → 중복 `_` 제거

```python
def slugify_column(name: str) -> str:
    value = name.strip().lower()
    for source, target in (("%", "percent"), ("/", "_per_"), ("-", "_"), (" ", "_")):
        value = value.replace(source, target)
    value = re.sub(r"[^a-z0-9_]", "_", value)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")
```

| 원본 CSV 헤더 | 정규화 키 | record 포함 여부 |
|---|---|---|
| Network Slice ID | `network_slice_id` | ✅ 추적용 |
| Timestamp | `timestamp` | ✅ 추적용 |
| Device ID | `device_id` | ✅ 추적용 |
| Traffic Type | `traffic_type` | ✅ QoS 평가 보조 |
| Network Utilization | `network_utilization_percent` | ✅ KPI |
| Latency ms | `latency_ms` | ✅ KPI |
| Packet Loss Rate | `packet_loss_rate_percent` | ✅ KPI |
| Signal Strength dBm | `signal_strength_dbm` | ✅ KPI |
| Bandwidth Utilization | `bandwidth_utilization_percent` | ✅ KPI |
| QoS Metric Throughput | `qos_metric_throughput` | ✅ KPI |
| Overload Status | `overload_status` | ✅ KPI |
| Network Slice Failure | `network_slice_failure` | ❌ **비표준 KPI** |
| Device Type | `device_type` | ❌ **비표준 KPI** |
| Region | `region` | ❌ **비표준 KPI** |
| Network Failure Count | `network_failure_count` | ❌ **비표준 KPI** |
| Time of Day | `time_of_day` | ❌ **비표준 KPI** |
| Weather Conditions | `weather_conditions` | ❌ **비표준 KPI** |

**비표준 KPI 제거 이유**: 3GPP TS 23.501 표준 KPI가 아니거나 시뮬레이션 아티팩트로 신뢰도가 낮다고 판정.  
JSONL `metadata`에도 포함되지 않으므로 모델이 간접 참조하는 경로 자체를 차단한다.

---

## 4. `build_record()` — metadata 구성

```python
def build_record(index: int, row: dict[str, str], normalized_columns: dict[str, str]) -> dict:
    normalized = normalize_row(row, normalized_columns)

    record = {
        "network_slice_id": normalized.get("network_slice_id", "0"),
        "timestamp":        normalized.get("timestamp", ""),
        "device_id":        normalized.get("device_id", "0"),

        "traffic_type":                  coerce_int(normalized.get("traffic_type", "0")),
        "network_utilization_percent":   coerce_float(normalized.get("network_utilization_percent", "0")),
        "latency_ms":                    coerce_float(normalized.get("latency_ms", "0")),
        "packet_loss_rate_percent":      coerce_float(normalized.get("packet_loss_rate_percent", "0")),
        "signal_strength_dbm":           coerce_float(normalized.get("signal_strength_dbm", "0")),
        "bandwidth_utilization_percent": coerce_float(normalized.get("bandwidth_utilization_percent", "0")),
        "qos_metric_throughput":         coerce_float(normalized.get("qos_metric_throughput", "0")),
        "overload_status":               coerce_int(normalized.get("overload_status", "0")),
    }

    return {
        "id":          f"network-slicing-{index:05d}",
        "domain":      "network_slicing_qos",
        "system":      SYSTEM_PROMPT,
        "instruction": build_instruction(record),
        "input":       "",
        "output":      build_expected_output(record),
        "metadata":    record,
    }
```

**record 저장 형식**:

| 필드 | 저장 형식 | 비고 |
|---|---|---|
| `network_slice_id`, `timestamp`, `device_id` | str (원본) | 추적용, 분류에 미사용 |
| `traffic_type` | int (0/1/2/3) | `_get_sst_label()` / `_get_sla_thresholds()` 입력 |
| KPI 7개 (`latency_ms` 등) | float **0~1 정규화** | 물리값은 `_restore_*()` 호출로 복원 |
| `overload_status` | int (0 또는 1) | 정규화 없이 그대로 사용 |

---

## 5. 정규화 값 → 물리 값 복원 함수

record에 저장된 값은 Min-Max 정규화 상태다.  
분류·플래그 계산·instruction 생성 시 전용 복원 함수로 물리 단위로 환산한다.

```python
def _restore_latency_ms(normalized_latency: float) -> float:
    return normalized_latency * 495.0 + 5.0          # 5 ms ~ 500 ms

def _restore_packet_loss_percent(normalized_loss: float) -> float:
    return normalized_loss * 100.0                   # 0% ~ 100%

def _restore_throughput_percent(normalized_throughput: float) -> float:
    return normalized_throughput * 100.0             # 0% ~ 100%

def _restore_util_percent(normalized_util: float) -> float:
    return normalized_util * 100.0                   # 0% ~ 100%

def _restore_signal_rsrp_dbm(normalized_signal: float) -> float:
    return -156.0 + normalized_signal * 125.0        # -156 dBm ~ -31 dBm
```

| 필드 | 복원 공식 | 물리 범위 |
|---|---|---|
| `latency_ms` | `x × 495 + 5` | 5 ms ~ 500 ms |
| `packet_loss_rate_percent` | `x × 100` | 0% ~ 100% |
| `qos_metric_throughput` | `x × 100` | 0% ~ 100% |
| `network_utilization_percent` | `x × 100` | 0% ~ 100% |
| `bandwidth_utilization_percent` | `x × 100` | 0% ~ 100% |
| `signal_strength_dbm` | `-156 + x × 125` | -156 dBm ~ -31 dBm |
| `overload_status` | (없음, 그대로 사용) | 0 또는 1 |

---

## 6. `traffic_type` → 3GPP SLA 기준값

```python
def _get_sla_thresholds(traffic_type: int) -> tuple[float, float]:
    if traffic_type == 2:
        return 300.0, 0.01   # eMBB-video (5QI=4)
    if traffic_type == 3:
        return 10.0, 0.01    # URLLC (5QI=82)
    return 100.0, 1.0        # eMBB-voice (5QI=1), traffic_type 0/1

def _get_sst_label(traffic_type: int) -> str:
    if traffic_type == 2:
        return "eMBB-video"
    if traffic_type == 3:
        return "URLLC"
    return "eMBB-voice"
```

| `traffic_type` | 레이블 | PDB (ms) | PER (%) | 3GPP 5QI |
|---|---|---|---|---|
| 0, 1 | eMBB-voice | 100 ms | 1.0% | 5QI=1 |
| 2 | eMBB-video | 300 ms | 0.01% | 5QI=4 |
| 3 | URLLC | 10 ms | 0.01% | 5QI=82 |

PDB (Packet Delay Budget): 지연 허용 상한.  
PER (Packet Error Rate): 패킷 손실 허용 상한.

---

## 7. `_get_prompt_flags()` — boolean 플래그 6개

소형 1B 모델은 `39.15 > 1.0` 같은 수치 비교를 신뢰할 수 없다.  
Python rule engine이 미리 계산한 `yes/no` 플래그를 instruction에 주입해서 모델이 "읽고 복사"하도록 유도한다.

```python
def _get_prompt_flags(record: dict) -> dict[str, str]:
    traffic_type   = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)

    latency_ms     = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct       = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    throughput_pct = _restore_throughput_percent(float(record["qos_metric_throughput"]))
    net_util_pct   = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct    = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm       = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))
    overload       = int(record["overload_status"])

    latency_exceeds_pdb     = latency_ms > pdb_ms
    packet_loss_exceeds_per = loss_pct > per_pct
    packet_loss_abnormal    = loss_pct >= 70.0
    signal_critical         = rsrp_dbm < -110.0

    hard_breach = (
        latency_exceeds_pdb
        or loss_pct > 5.0
        or packet_loss_abnormal
        or signal_critical
        or (overload == 1 and net_util_pct >= 80.0
            and (latency_ms > pdb_ms or loss_pct > per_pct))
    )

    stable_allowed = (
        not hard_breach
        and latency_ms <= pdb_ms
        and loss_pct <= per_pct
        and overload == 0
        and net_util_pct < 80.0
        and bw_util_pct < 80.0
        and rsrp_dbm >= -100.0
        and throughput_pct >= 40.0
    )

    return {
        "latency_exceeds_pdb":     "yes" if latency_exceeds_pdb else "no",
        "packet_loss_exceeds_per": "yes" if packet_loss_exceeds_per else "no",
        "packet_loss_abnormal":    "yes" if packet_loss_abnormal else "no",
        "signal_critical":         "yes" if signal_critical else "no",
        "hard_breach":             "yes" if hard_breach else "no",
        "stable_allowed":          "yes" if stable_allowed else "no",
    }
```

| 플래그 | 조건 | 의미 |
|---|---|---|
| `latency_exceeds_pdb` | `latency_ms > pdb_ms` | 지연 SLA 위반 여부 |
| `packet_loss_exceeds_per` | `loss_pct > per_pct` | 패킷 손실 SLA 위반 여부 |
| `packet_loss_abnormal` | `loss_pct >= 70.0%` | 비정상 손실 (시뮬레이션 아티팩트 가능성) |
| `signal_critical` | `rsrp_dbm < -110.0 dBm` | 서비스 불가 신호 강도 |
| `hard_breach` | 위 4개 중 하나 OR 복합 overload | **즉시 critical 강제 집합 플래그** |
| `stable_allowed` | 모든 위험 조건 불만족 AND 정상 범위 전부 충족 | stable 출력 허용 여부 |

**`hard_breach` 구성 논리**:

```
hard_breach = (
    latency > pdb                                    — PDB 직접 위반
    OR loss > 5%                                     — PER을 크게 초과
    OR loss >= 70%                                   — 비정상(시뮬레이션 아티팩트)
    OR rsrp < -110 dBm                               — 서비스 불가 신호
    OR (overload=1 AND net_util≥80%
        AND (latency > pdb OR loss > per))           — 과부하+자원포화+SLA 위반 복합
)
```

**`stable_allowed`** 는 `NOT hard_breach`의 단순 역이 아닌, 더 엄격한 전항 조건:

```
stable_allowed = (
    NOT hard_breach
    AND latency ≤ pdb
    AND loss ≤ per
    AND overload = 0
    AND net_util < 80%
    AND bw_util < 80%
    AND rsrp ≥ -100 dBm
    AND throughput ≥ 40%
)
```

---

## 8. `classify_qos_state()` — QoS 상태 분류

규칙 기반 라벨 엔진. 복원된 물리 값을 SLA 기준과 직접 비교해 `critical / degraded / stable` 중 하나를 반환한다.

```python
def classify_qos_state(record: dict) -> str:
    latency_ms     = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct       = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    throughput_pct = _restore_throughput_percent(float(record["qos_metric_throughput"]))
    net_util_pct   = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct    = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm       = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))
    overload       = int(record["overload_status"])
    traffic_type   = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)

    # ── critical (해당 시 즉시 반환) ──────────
    if loss_pct >= 70.0:
        return "critical"
    if latency_ms > pdb_ms:
        return "critical"
    if loss_pct > 5.0:
        return "critical"
    if rsrp_dbm < -110.0:
        return "critical"
    if overload == 1 and net_util_pct >= 80.0 and (latency_ms > pdb_ms or loss_pct > per_pct):
        return "critical"

    # ── degraded ────────────────────────────
    if pdb_ms * 0.8 < latency_ms <= pdb_ms:
        return "degraded"
    if per_pct < loss_pct < 5.0:
        return "degraded"
    if 80.0 <= net_util_pct < 90.0:
        return "degraded"
    if 80.0 <= bw_util_pct < 90.0:
        return "degraded"
    if overload == 1:
        return "degraded"
    if -110.0 <= rsrp_dbm < -100.0:
        return "degraded"
    if throughput_pct < 40.0:
        return "degraded"

    # ── stable ──────────────────────────────
    if (
        latency_ms <= pdb_ms
        and loss_pct <= per_pct
        and overload == 0
        and net_util_pct < 80.0
        and bw_util_pct < 80.0
        and rsrp_dbm >= -100.0
        and throughput_pct >= 40.0
    ):
        return "stable"

    return "degraded"
```

### critical 조건 표 (우선순위 순)

| 우선순위 | 조건 | 근거 |
|---|---|---|
| 1 | `loss_pct >= 70.0%` | 비정상 손실 — 시뮬레이션 아티팩트, 보수적 critical 처리 |
| 2 | `latency_ms > pdb_ms` | TS 23.501 PDB 직접 위반 |
| 3 | `loss_pct > 5.0%` | PER 기준을 크게 초과한 심각 손실 |
| 4 | `rsrp_dbm < -110.0 dBm` | 서비스 불가 수준 신호 강도 |
| 5 | `overload=1 AND net_util≥80% AND (latency>pdb OR loss>per)` | 과부하+자원포화+SLA 위반 복합 |

### degraded 조건 표

| 조건 | 근거 |
|---|---|
| `pdb×0.8 < latency ≤ pdb` | PDB의 80~100% 구간 — 위반 직전 경고 |
| `per_pct < loss < 5.0%` | PER 초과이나 심각 수준 미만 |
| `80% ≤ net_util < 90%` | 자원 포화 임박 |
| `80% ≤ bw_util < 90%` | 대역폭 포화 임박 |
| `overload = 1` | 과부하 플래그 (위 임계 미도달 상태) |
| `-110 ≤ rsrp < -100 dBm` | 신호 불량 구간 |
| `throughput < 40%` | 처리량 부족 |

---

## 9. `recommend_action()` — 운영 액션 생성

```python
def recommend_action(record: dict, qos_state: str) -> str:
    # (물리 값 복원 코드 생략 — classify_qos_state()와 동일)

    if loss_pct >= 70.0:
        return "isolate the slice, validate telemetry, and reroute traffic immediately"
    if overload == 1 and net_util_pct >= 80.0 and (latency_ms > pdb_ms or loss_pct > per_pct):
        return "redistribute slice traffic and reduce utilization below 80 percent"
    if latency_ms > pdb_ms:
        return "prioritize low-latency scheduling and reduce contention"
    if loss_pct > 5.0:
        return "investigate radio quality and reduce packet loss with retransmission tuning"
    if rsrp_dbm < -110.0:
        return "trigger handover or beam management to recover signal quality"
    if overload == 1 or net_util_pct >= 80.0 or bw_util_pct >= 80.0:
        return "rebalance load and scale resources before SLA violation worsens"
    if loss_pct > per_pct:
        return "tune QoS handling to bring packet loss back within PER"
    if -110.0 <= rsrp_dbm < -100.0:
        return "improve RF conditions and mobility control to raise signal strength"
    if throughput_pct < 40.0:
        return "review scheduling and bandwidth allocation to recover throughput"
    if qos_state == "stable":
        return "maintain the current policy and continue KPI monitoring"
    return "review slice resource allocation and monitor KPI trends closely"
```

| 조건 (우선순위)                      | 의미            | 운영 액션           |
| ------------------------------ | ------------- | --------------- |
| loss ≥ 70%                     | 네트워크 붕괴 수준 손실 | 슬라이스 격리 + 즉시 우회 |
| overload + util ≥ 80% + SLA 위반 | 리소스 포화 상태     | 트래픽 재분배         |
| latency > PDB                  | 지연 SLA 위반     | 저지연 스케줄링 적용     |
| loss > 5%                      | 심각한 손실        | RF 점검 + 재전송 튜닝  |
| RSRP < -110                    | 신호 붕괴 수준      | 핸드오버 / 빔 관리     |
| overload or util ≥ 80%         | 과부하 전조        | 부하 분산           |
| loss > PER                     | 경미한 SLA 위반    | QoS 정책 튜닝       |
| -110 ≤ RSRP < -100             | 신호 저하         | RF 환경 개선        |
| throughput < 40%               | 처리량 부족        | 스케줄링 재조정        |
| stable                         | 정상 상태         | 유지              |
| 기타                             | 애매한 저하        | 자원 재검토          |


---

## 10. `build_instruction()` — LLM 입력 생성

`hard_breach`와 `stable_allowed`를 맨 앞에 배치해 모델이 가장 먼저 결정적 분류 단서를 보도록 한다.

```python
def build_instruction(record: dict) -> str:
    traffic_type = int(record["traffic_type"])
    sst_label    = _get_sst_label(traffic_type)
    flags        = _get_prompt_flags(record)
    overload     = int(record["overload_status"])

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
```

---

## 11. `build_expected_output()` — 정답 생성

```python
def build_expected_output(record: dict) -> str:
    traffic_type = int(record["traffic_type"])
    pdb_ms, per_pct = _get_sla_thresholds(traffic_type)
    sst_label    = _get_sst_label(traffic_type)

    latency_ms   = _restore_latency_ms(float(record["latency_ms"]))
    loss_pct     = _restore_packet_loss_percent(float(record["packet_loss_rate_percent"]))
    net_util_pct = _restore_util_percent(float(record["network_utilization_percent"]))
    bw_util_pct  = _restore_util_percent(float(record["bandwidth_utilization_percent"]))
    rsrp_dbm     = _restore_signal_rsrp_dbm(float(record["signal_strength_dbm"]))

    qos_state = classify_qos_state(record)
    action    = recommend_action(record, qos_state)

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
```

**실제 출력 예시**:

```
QoS state: critical
Reason: latency 369.6ms exceeds PDB 100ms for eMBB-voice
Action: prioritize low-latency scheduling and reduce contention
```

```
QoS state: critical
Reason: packet loss 93.9% is abnormal and exceeds PER 0.01%
Action: isolate the slice, validate telemetry, and reroute traffic immediately
```

**Reason 우선순위**: 비정상 손실 → latency 위반 → 심각 손실 → 신호 위험 → degraded 종합 → stable 정상.

---

## 12. `SYSTEM_PROMPT` — 현재 코드 전문 (20개 few-shot)

```python
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
```

**SYSTEM_PROMPT 구성 요소 설명**:

| 요소 | 내용 | 의도 |
|---|---|---|
| 출력 형식 선언 | `Return exactly 3 lines` | 3줄 고정 구조 강제 |
| Mandatory rules | `hard_breach=yes → critical` 등 7개 명시 규칙 | 소형 모델이 플래그를 무시하는 경향 방지 |
| `First decide... Then write...` | 2단계 생성 지시 | 분류 결정 → Reason/Action 순서 명시 |
| 20개 few-shot examples | critical 13개 + degraded 4개 + stable 3개 | 레이블별 다양한 케이스 패턴 주입 |

**few-shot 구성비**:

| 레이블 | 개수 | 비율 |
|---|---|---|
| critical | 13개 (Ex 1~12, 20) | 65% |
| degraded | 4개 (Ex 13~16) | 20% |
| stable | 3개 (Ex 17~19) | 15% |

---

## 13. JSONL 파일 구조 및 실제 샘플

**파일 위치**: `jetson_slm_stack/dataset/prepared/network_slicing_qos/`

| 파일 | 샘플 수 | 용도 |
|---|---|---|
| `train.jsonl` | 400건 | 파인튜닝 학습용 |
| `val.jsonl` | 50건 | 학습 중 과적합 모니터링용 |
| `test.jsonl` | 50건 | 최종 성능 평가용 (`test_dataset.sh`에서 사용) |
| `manifest.prep.json` | — | 데이터셋 메타정보 (CSV 경로·컬럼·split 비율·생성 시각 등) |
| `csv_schema.json` | — | 원본 컬럼 → 정규화 키 매핑 |
| `csv_preview.json` | — | 첫 5행 미리보기 |

**JSONL 1줄 실제 구조**:

```json
{
  "id": "network-slicing-00009",
  "domain": "network_slicing_qos",
  "system": "<SYSTEM_PROMPT — 20개 few-shot 포함>",
  "instruction": "traffic=eMBB-voice; overload=0; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; packet_loss_abnormal=no; signal_critical=no; hard_breach=yes; stable_allowed=no",
  "input": "",
  "output": "QoS state: critical\nReason: latency 369.6ms exceeds PDB 100ms for eMBB-voice\nAction: prioritize low-latency scheduling and reduce contention",
  "metadata": {
    "network_slice_id": "...",
    "timestamp": "...",
    "device_id": "...",
    "traffic_type": 0,
    "network_utilization_percent": 0.742,
    "latency_ms": 0.737,
    "packet_loss_rate_percent": 0.369,
    "signal_strength_dbm": 0.483,
    "bandwidth_utilization_percent": 0.651,
    "qos_metric_throughput": 0.612,
    "overload_status": 0
  }
}
```

**실제 test.jsonl 샘플 3건 (head 확인 결과)**:

| id | instruction (요약) | expected output (첫 줄) |
|---|---|---|
| network-slicing-00009 | traffic=eMBB-voice; overload=0; ...; hard_breach=yes; stable_allowed=no | QoS state: critical |
| network-slicing-00019 | traffic=eMBB-video; overload=1; packet_loss_abnormal=yes; hard_breach=yes | QoS state: critical |
| network-slicing-00029 | traffic=eMBB-voice; overload=1; latency_exceeds_pdb=yes; hard_breach=yes | QoS state: critical |

## Input 지표 정의 및 판정 메커니즘  

### 1. 주요 입력 지표 (Input Features)
각 지표는 네트워크의 현재 상태를 결정하는 핵심 성능 지표(KPI)입니다.

* **`hard_breach` (SLA 위반 여부)**
    * 서비스 수준 협약(SLA)이 완전히 깨졌는지 나타내는 가장 핵심 지표 (yes/no)
* **`stable_allowed` (안정 유지 가능성)**
    * 현재 시스템이 안정(Stable) 상태를 유지할 수 있는 환경인지 여부 (yes/no)
* **`packet_loss_abnormal` (비정상 패킷 손실)**
    * 단순 부하가 아닌 하드웨어/시스템 오류로 인한 돌발적 유실 여부 (yes/no)
* **`signal_critical` (신호 품질 위기)**
    * 무선 신호 세기가 서비스 가능 임계값 미만으로 저하되었는지 여부 (yes/no)
* **`latency_exceeds_pdb` (지연 시간 초과)**
    * 허용 지연 시간(PDB, Packet Delay Budget)을 초과했는지 여부 (yes/no)
* **`packet_loss_exceeds_per` (패킷 오류율 초과)**
    * 허용 패킷 오류율(PER, Packet Error Rate)을 초과했는지 여부 (yes/no)
* **`traffic` (서비스 유형)**
    * 트래픽의 종류 및 요구사항 (eMBB-voice, eMBB-video, URLLC 등)
* **`overload` (시스템 과부하)**
    * 자원 사용량이 임계치를 넘었는지 나타내는 상태 값 (1: 과부하, 0: 정상)

### 2. 상태 판정 로직 요약 (Logic Summary)

* **🔴 Critical (심각)**
    * `hard_breach=yes` 상태가 기반이 됨
    * 지연(`latency`), 손실(`packet_loss`), 신호(`signal`) 중 하나라도 `yes`일 때 판정
* **🟡 Degraded (저하)**
    * `hard_breach=no` 이지만 `overload=1`이거나 지표가 불안정할 때 판정
* **🟢 Stable (안정)**
    * `hard_breach=no` 이며 모든 KPI가 정상 범위 내에 있을 때 판정


### 3. 상황별 대응 프로세스 (Action Flow)
1.  **지연 문제 발생 시:** 저지연 스케줄링 우선순위 부여 및 PRB 경합 완화
2.  **신호 불량 발생 시:** 핸드오버(Handover) 또는 빔 관리(Beam Management) 실행
3.  **시스템 오류 발생 시:** 슬라이스 격리(Isolate), 텔레메트리 검증 및 트래픽 우회
4.  **단순 과부하 발생 시:** 부하 재분산(Load Balancing) 및 가동률 80% 이하 유지
   
---

## 14. Split 정책 및 생성 결과

```python
SPLIT_RULE = {"train": 8, "val": 1, "test": 1}
MAX_SAMPLES = 50   # 전체 처리 제한

def choose_split(index: int) -> str:
    bucket = index % 10
    if bucket < SPLIT_RULE["train"]:      return "train"  # 0~7
    if bucket < SPLIT_RULE["train"] + SPLIT_RULE["val"]: return "val"   # 8
    return "test"                                                         # 9
```

**실제 split 결과** (`manifest.prep.json` 기록):

| split | index % 10 | 샘플 수 |
|---|---|---|
| train | 0 ~ 7 | **400건** |
| val | 8 | **50건** |
| test | 9 | **50건** |
| 합계 | — | **500건** |

> `MAX_SAMPLES = 50` 설정에도 불구하고 500건이 생성되었다.  
> 코드 내 `if total >= MAX_SAMPLES: break` 조건이 실제로는 동작하지 않은 것으로 보인다.

**deterministic 분할 특성**:
- 동일 CSV + 동일 row 순서 → 항상 동일한 split 결과
- 무작위 분할 대비 재현성·디버깅·비교 실험에 유리
- **한계**: row 순서가 편향되어 있으면 그 편향이 split에 그대로 전파됨

---

## 15. 평가 환경 구성

### `.env` (jetson_slm_stack/.env) 핵심 설정

```ini
# 디바이스 / 정밀도
DEVICE=cuda
DTYPE=float16
GPU_OFFLOAD_ENABLED=1
GPU_TARGET_MEMORY_MB=896
GPU_MEMORY_RESERVE_MB=768

# 토큰 한도
MAX_INPUT_TOKENS=448
MAX_NEW_TOKENS=256

# 생성 파라미터
TEMPERATURE=0.1
TOP_P=0.9
TOP_K=30
REPETITION_PENALTY=1.10

# 기타
ENABLE_WARMUP=1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
```

### `test_dataset.sh` 실행 방법

```bash
# test split 전체 평가
bash test_dataset.sh llama --api chat --split test

# 최대 30건
bash test_dataset.sh llama --api chat --split test --max 30

# val split 검증
bash test_dataset.sh llama --api chat --split val
```

채팅 API 호출 시 `messages` 배열은 **system(SYSTEM_PROMPT) + user(instruction) 2개**로만 구성된다.  
별도 few-shot 쌍은 포함하지 않으며, 20개 few-shot은 SYSTEM_PROMPT(`system` 필드) 내에 텍스트로 포함되어 전달된다.  
`stop: ["Example ", "\n\n\n"]` 토큰으로 모델이 few-shot 패턴을 무한 반복하는 현상을 차단한다.

---

## 16. 평가 결과 — 2026-03-30 (test split 9건, messages 구조 변경 후)

**실행 명령**:
```bash
bash test_dataset.sh llama --split test --api chat --max 30
```

**결과 파일**: `test_slm_output/dataset_eval_llama_test_20260330_003535.jsonl`

### 16-1. 정확도 요약

| 항목 | 이전 실행 (2026-03-29) | 이번 실행 (2026-03-30) |
|---|---|---|
| 평가 샘플 수 | 30건 (완료) | **9건 (중단)** |
| 기대 레이블 분포 | critical: 30개 (100%) | critical: 9개 (100%) |
| 모델 출력 분포 | degraded: 30개 (100%) | **degraded: 9개 (100%)** |
| **QoS state 정확도** | **0 / 30 = 0.0%** | **0 / 9 = 0.0%** |

### 16-2. 성능 지표

| 지표 | 이전 (2026-03-29) | 이번 (2026-03-30) | 변화 |
|---|---|---|---|
| `prompt_tokens` | 508 tokens | **331 tokens** | ▼ −177 (truncation 해소) |
| `completion_tokens` | 34 tokens | **30.2 tokens** | ≈ 동일 |
| `latency_sec` | 53.6 sec/req | **18.66 sec/req** | ▼ −65% |
| `tokens_per_sec` | 0.635 tok/s | **1.621 tok/s** | ▲ +2.6× |

### 16-3. 샘플별 상세 (3건)

**Sample 1 — network-slicing-00009**

```
instruction  traffic=eMBB-voice; overload=0; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes;
             packet_loss_abnormal=no; signal_critical=no; hard_breach=yes; stable_allowed=no

expected     QoS state: critical
             Reason: latency 369.6ms exceeds PDB 100ms for eMBB-voice
             Action: prioritize low-latency scheduling and reduce contention

generated    QoS state: degraded
             Reason: latency exceeds PDB and packet loss exceeds PER for eMBB-voice
             Action: increase traffic shaping to limit bandwidth usage

metrics      prompt_tokens=332, completion_tokens=34, latency=21.071s, tps=1.614
```

**Sample 2 — network-slicing-00019**

```
instruction  traffic=eMBB-video; overload=1; latency_exceeds_pdb=no; packet_loss_exceeds_per=yes;
             packet_loss_abnormal=yes; signal_critical=no; hard_breach=yes; stable_allowed=no

expected     QoS state: critical
             Reason: packet loss 93.9% is abnormal and exceeds PER 0.01%
             Action: isolate the slice, validate telemetry, and reroute traffic immediately

generated    QoS state: degraded
             Reason: packet loss exceeds PER for eMBB-video
             Action: increase bandwidth allocation to improve video quality

metrics      prompt_tokens=331, completion_tokens=28, latency=17.404s, tps=1.609
```

**Sample 3 — network-slicing-00029**

```
instruction  traffic=eMBB-voice; overload=1; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes;
             packet_loss_abnormal=no; signal_critical=no; hard_breach=yes; stable_allowed=no

expected     QoS state: critical
             Reason: latency 338.0ms exceeds PDB 100ms for eMBB-voice
             Action: prioritize low-latency scheduling and reduce contention

generated    QoS state: degraded
             Reason: latency exceeds PDB and packet loss exceeds PER for eMBB-voice
             Action: increase traffic shaping to limit bandwidth usage

metrics      prompt_tokens=332, completion_tokens=34, latency=20.864s, tps=1.630
```

---

## A. Few-shot
### 🚀 Few-Shot 판정 가이드라인

### 1. 핵심 판정 로직 (Logic)
AI 모델의 상태 분류 및 대응을 위한 3대 핵심 규칙입니다.

* **🔴 Critical (심각)**
  - 조건: `hard_breach=yes` (SLA 위반 발생)
  - 트리거: 지연(PDB), 손실(PER), 신호(Signal) 중 1개 이상 지표 미달 시
* **🟡 Degraded (저하)**
  - 조건: `hard_breach=no` (치명적 위반 없음)
  - 트리거: 시스템 과부하(`overload=1`) 또는 KPI 지표가 경고 범위에 근접 시
* **🟢 Stable (안정)**
  - 조건: `hard_breach=no` 및 모든 KPI 수치가 정상 범위 내 존재
  - 트리거: 안정적 운영 환경(`stable_allowed=yes`) 확보 시

---

### 2. 주요 Few-Shot 예시 (Examples)

#### **[Ex 1] 복합 장애 상황 (Critical)**
* **Input:** `hard_breach=yes; latency_exceeds_pdb=yes; packet_loss_exceeds_per=yes; traffic=eMBB-voice`
* **QoS State:** **심각 (Critical)**
* **Reason:** eMBB-voice 지연 시간(PDB) 및 패킷 손실(PER) 허용치 초과
* **Action:** 저지연 스케줄링 우선순위 상향 및 PRB 자원 경합 완화

#### **[Ex 13] 과부하 발생 상황 (Degraded)**
* **Input:** `hard_breach=no; stable_allowed=no; overload=1; traffic=eMBB-voice`
* **QoS State:** **저하 (Degraded)**
* **Reason:** 치명적 위반은 없으나 시스템 과부하 발생
* **Action:** 부하 재분산(Load Balancing) 및 KPI 추이 정밀 모니터링

#### **[Ex 17] 정상 운영 상황 (Stable)**
* **Input:** `hard_breach=no; stable_allowed=yes; overload=0; traffic=eMBB-voice`
* **QoS State:** **안정 (Stable)**
* **Reason:** 모든 KPI 수치가 정상 범위 내 위치
* **Action:** 현재 슬라이스 정책 유지 및 모니터링 지속

---

### 💡 학습 포인트
* **의사결정 구조:** `hard_breach` 여부에 따른 상태 분기 학습
* **조치 최적화:** 장애 원인(지연, 손실, 과부하 등)에 최적화된 Action 매칭
* **문맥 파악:** 서비스 유형(`traffic`)별 요구사항에 따른 유연한 판정 유도

---

## 17. 실패 원인 분석

### 17-1. 핵심 증상

- 30개 전부 `QoS state: degraded` 출력 — `hard_breach=yes` 완전 무시
- `packet_loss_abnormal=yes` 케이스 6건 모두 무시 → degraded 출력
- `signal_critical=yes` 케이스 7건 모두 무시 → degraded 출력
- `signal_critical=yes` AND `packet_loss_abnormal=yes` 동시 케이스(1건)도 무시
- `Reason` 문장은 SYSTEM_PROMPT few-shot과 유사하게 생성 → **3줄 출력 형식 자체는 100% 올바름**
- `Action` 일부는 정답과 완전 일치 → QoS **레이블 결정만** 실패
- messages에서 별도 few-shot 2쌍 제거 후에도 동일한 패턴 지속 → `.sh` few-shot 쌍은 근본 원인이 아니었음

### 17-2. 원인 1 — instruction 포맷과 few-shot 포맷 불일치 (가장 유력한 원인)

| 구분 | 필드 순서 |
|---|---|
| 현재 JSONL `instruction` (평가 시 실제 입력) | `traffic=...; overload=...; ...; hard_breach=...; stable_allowed=...` ← **traffic 선두, hard_breach 후미** |
| SYSTEM_PROMPT `FEW_SHOT_EXAMPLES` | `hard_breach=...; stable_allowed=...; ...; traffic=...; overload=...` ← **hard_breach 선두** |

모델이 few-shot에서 학습한 "hard_breach=yes → 첫 번째 키 → QoS state: critical" 연결 고리가  
실제 입력(hard_breach가 7번째 키)에서는 위치적으로 동작하지 않는다.  
→ **JSONL 재생성 후 재평가 필요**

### 17-3. 원인 2 — 1B 모델의 명령 추종 한계

1B 규모 모델은 "Mandatory rules: hard_breach=yes → critical" 같은 텍스트 지시를  
prior next-token probability 앞에서 억제하지 못하는 경향이 있다.  
즉, few-shot 패턴이 `degraded`로 수렴되면 규칙을 무시하고 degraded를 출력한다.

### 17-4. 속도

| 지표 | 이전 (messages 3쌍) | 이번 (messages 2개) | 기대값 |
|---|---|---|---|
| 평균 latency | 53.6 sec/req | **18.53 sec/req** | DGX 기준 1~5 sec 예상 |
| tokens_per_sec | 0.635 tok/s | **1.63 tok/s** | DGX 기준 50~100+ tok/s 예상 |

→ messages 축소로 속도가 2.6× 개선되었으나 여전히 DGX 기대치 대비 느림. GPU offload·float16 설정 재확인 필요.

---

### 17-5. 평가 요약 — 되는 것 / 안 되는 것

#### ✅ 되는 것 (모델이 올바르게 수행한 항목)

| 항목 | 결과 | 근거 |
|---|---|---|
| **3줄 출력 형식 준수** | 30/30 (100%) | `QoS state:` / `Reason:` / `Action:` 3줄 구조 완벽 출력 |
| **Reason 문장 생성** | 30/30 (100%) | traffic 유형(eMBB-voice, URLLC 등) 올바르게 포함한 자연어 문장 생성 |
| **Action 문장 생성** | 30/30 (100%) | 도메인에 적합한 운영 액션 문장 생성 |
| **stop token 준수** | 30/30 (100%) | `"Example "` 토큰에서 생성 중단, few-shot 패턴 반복 없음 |
| **instruction 필드 파싱** | 30/30 (100%) | `traffic=`, `overload=` 등 key-value 필드 인식 |

#### ❌ 안 되는 것 (모델이 올바르게 수행하지 못한 항목)

| 항목 | 결과 | 근거 |
|---|---|---|
| **`hard_breach=yes` → critical 규칙** | 0/30 (0%) | 30건 전부 `degraded` 출력, SYSTEM_PROMPT의 Mandatory rules 무시 |
| **`packet_loss_abnormal=yes` → critical 규칙** | 0/6 (0%) | 6건 전부 무시, 이 플래그도 분류에 영향 없음 |
| **`signal_critical=yes` → critical 규칙** | 0/7 (0%) | 7건 전부 무시, 가장 강한 조건도 무시 |
| **복합 위반(signal(신호 상태)+packet(패킷 손실) --> 동시) → critical** | 0/1 (0%) | 두 플래그 동시 yes여도 무시 |
| **QoS state 레이블 결정 정확도** | 0/30 (0%) | 모든 경우에서 `degraded`로 수렴 |
| **`stable` 레이블 출력** | — | test split에 stable 기대 샘플 없어 미검증 (별도 평가 필요) |

---
#### 복합 위반 부연 설명
- 복합 위반인데도 critical로 못 감 = rule 무시

##### signal
> signal_critical = rsrp_dbm < -110.0
- (무선 신호가 임계값 이하 → 통신 품질 매우 나쁨)

##### packet
> packet_loss_abnormal = loss_pct >= 70.0
packet_loss_exceeds_per = loss_pct > per_pct
- 패킷 손실률이 비정상적으로 높음 → 데이터 전달 실패
