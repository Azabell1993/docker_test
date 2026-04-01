"""
network_traffic_dataset.py
==========================
5G Network Traffic QoS/QoE Diagnosis Engine
--------------------------------------------
Dataset : jetson_slm_stack/dataset/raw/network_traffic/5g_network_data.csv
Author  : refactored from network_slicing_qos pipeline

Refactor summary
----------------
REMOVED:
  - Network-slicing-specific constructs (_get_sla_thresholds, _get_sst_label,
    traffic_type 1/2/3 mapping)
  - 0-to-1 normalization restoration functions (_restore_latency_ms, etc.)
    -- new CSV contains raw physical-unit values; no restore step needed.
  - FEW_SHOT_EXAMPLES / SYSTEM_PROMPT and all LLM JSONL generation logic
    (build_instruction, build_record, build_expected_output, convert_csv_to_jsonl)
  - DATASET_SPEC references to Kaggle 6G slicing dataset
  - MAX_SAMPLES = 50 hard limit; pipeline now processes the full dataset

MODIFIED:
  - Path setup -> points to raw/network_traffic
  - slugify_column() -> preserved but superseded by explicit COLUMN_MAP
  - inspect_csv() -> simplified; full schema is known in advance
  - coerce_float / coerce_int -> replaced by pandas coercion
  - main() -> rebuilt around the 7-phase diagnosis pipeline

ADDED (new in this script):
  - COLUMN_MAP           : raw CSV header -> internal snake_case
  - KPI_META             : 3GPP-inspired interpretation context per KPI
  - KPI_THRESHOLDS       : centralized normal / warning / critical bounds
  - DEVICE_THRESHOLDS    : secondary device-side factors (battery, temperature)
  - SeverityLevel enum   : NORMAL / WARNING / CRITICAL
  - RootCause enum       : radio / core / mobility / transport / device / ...
  - BreachedKPI          : dataclass for a single KPI violation
  - AttentionProxy       : explainability proxy (NOT transformer attention)
  - DiagnosisRecord      : per-record output dataclass
  - SummaryReport        : aggregated statistics dataclass
  - load_and_clean()     : Phase 1 -- pandas-based loading / type coercion
  - evaluate_kpi_threshold() : Phase 3 -- single-KPI threshold evaluation
  - evaluate_all_kpis()  : Phase 3 -- full-row KPI sweep
  - compute_composite_severity() : Phase 3 -- record-level severity
  - infer_root_cause()   : Phase 4 -- rule-based root-cause tree
  - recommend_action()   : Phase 5 -- structured action lookup table
  - compute_global_feature_importance() : Phase 6 -- RF or rule-based proxy
  - compute_local_rule_weights()         : Phase 6 -- per-record attention proxy
  - diagnose_record()    : Phase 6 -- end-to-end single-row diagnosis
  - build_summary_report()               : Phase 7 -- aggregate statistics
  - export_diagnosis_jsonl()             : Phase 7 -- per-record JSONL output
  - export_summary_json()                : Phase 7 -- summary JSON output
  - run_example_tests()  : illustrative validation on 3 hand-crafted rows
  - main()               : full pipeline entry point

Design constraints
------------------
- Rule-based + explainable ML baseline (no deep learning).
- Thresholds in physical units; no normalization that destroys interpretability.
- Each function is pure and testable (no side effects except I/O helpers).
- sklearn is optional; falls back to domain-expert rule weights if unavailable.
- "attention_proxy" is an explainability proxy:
    1. feature_importance_target_download  -- RF impurity or rule weights
    2. feature_importance_target_drop      -- RF impurity or rule weights
    3. local_rule_weights                  -- per-record rule firing weights
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import argparse
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

# -- scikit-learn is optional ------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not available -- ML-based feature importance disabled.",
        stacklevel=2,
    )

# -- safe_ops C++ extension (optional) ----------------------------------------
# Provides: configure_runtime(), get_cuda_mem_info(), probe_cuda_budget()
# Built from jetson_slm_stack/app/csrc/ via:
#   cd jetson_slm_stack/app/csrc && pip install -e .
# Falls back to Python/torch equivalents when not installed.
try:
    import safe_ops as _safe_ops  # type: ignore[import-not-found]
    _HAS_SAFE_OPS = True
except ImportError:
    _safe_ops = None              # type: ignore[assignment]
    _HAS_SAFE_OPS = False

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
RAW_DIR = DATASET_DIR / "raw" / "network_traffic"
RAW_CSV = RAW_DIR / "5g_network_data.csv"
OUTPUT_DIR = Path(os.getenv("DIAG_OUTPUT_DIR", str(DATASET_DIR / "prepared" / "network_traffic")))

# ============================================================
# Processing limits
# ============================================================
# MAX_SAMPLES : hard cap applied during a normal full run.
# TEST_SAMPLES: cap used when --test flag is passed.
# Both limits are applied BEFORE feature-importance fitting so the
# test path is fast and the full path stays within a predictable
# memory / runtime budget.
MAX_SAMPLES:  int = 500
TEST_SAMPLES: int = 10

# ============================================================
# LLM server defaults  (overridable via CLI flags)
# ============================================================
LLM_DEFAULT_HOST:       str = "127.0.0.1"
LLM_DEFAULT_PORT:       int = 8000   # Llama-1B is on port 8000 by default
LLM_DEFAULT_MAX_NEW_TOKENS: int = 256

# ============================================================
# Phase 1 -- Explicit column rename map
# ============================================================
# Maps raw CSV header strings (exactly as they appear) to internal
# snake_case names used throughout this module.
# ASSUMPTION: column names match 5g_network_data.csv headers verbatim.
# Update only this dict if the upstream CSV changes.
COLUMN_MAP: dict[str, str] = {
    "Timestamp":                "timestamp",
    "Location":                 "location",
    "Signal Strength (dBm)":    "signal_strength_dbm",
    "Download Speed (Mbps)":    "download_speed_mbps",
    "Upload Speed (Mbps)":      "upload_speed_mbps",
    "Latency (ms)":             "latency_ms",
    "Jitter (ms)":              "jitter_ms",
    "Network Type":             "network_type",
    "Device Model":             "device_model",
    "Carrier":                  "carrier",
    "Band":                     "band",
    "Battery Level (%)":        "battery_level_percent",
    "Temperature (°C)":         "temperature_celsius",
    "Connected Duration (min)": "connected_duration_min",
    "Handover Count":           "handover_count",
    "Data Usage (MB)":          "data_usage_mb",
    "Video Streaming Quality":  "video_streaming_quality",
    "VoNR Enabled":             "vonr_enabled",
    "Network Congestion Level": "network_congestion_level",
    "Ping to Google (ms)":      "ping_google_ms",
    "Dropped Connection":       "dropped_connection",
}

# ============================================================
# Phase 2 -- KPI metadata registry
# ============================================================
# Maps each diagnostic KPI to its 3GPP-inspired interpretation context.
# This metadata is used in report generation and to explain which layer
# each KPI belongs to (radio / core / mobility / transport / retainability).
#
# ASSUMPTIONS (clearly marked):
#   signal_strength_dbm -> RSRP-like proxy  [ASSUMPTION: no separate RSRP column]
#   latency_ms          -> E2E PDB-related  [ASSUMPTION: measures RAN+core+transport combined]
#   jitter_ms           -> PDV-like         [ASSUMPTION: single-hop sample, not PDV from 3GPP TS 23.501]
#   download/upload     -> throughput KPI   [ASSUMPTION: application-layer measurement]
#   network_congestion_level -> load proxy  [ASSUMPTION: categorical operator label, not PRB utilization]
#   dropped_connection  -> retainability    [ASSUMPTION: binary event flag per measurement interval]
KPI_META: dict[str, dict[str, str]] = {
    "signal_strength_dbm": {
        "domain":      "radio",
        "3gpp_analog": "RSRP (Reference Signal Received Power)",
        "unit":        "dBm",
        "direction":   "higher_is_better",
        "note":        "Proxy for UE radio coverage quality. Not RSRP from RRC measurements.",
    },
    "download_speed_mbps": {
        "domain":      "throughput",
        "3gpp_analog": "GBR / MBR downlink throughput",
        "unit":        "Mbps",
        "direction":   "higher_is_better",
        "note":        "DL user-plane throughput proxy; application-layer measurement.",
    },
    "upload_speed_mbps": {
        "domain":      "throughput",
        "3gpp_analog": "GBR / MBR uplink throughput",
        "unit":        "Mbps",
        "direction":   "higher_is_better",
        "note":        "UL user-plane throughput proxy; application-layer measurement.",
    },
    "latency_ms": {
        "domain":      "core",
        "3gpp_analog": "E2E Packet Delay Budget (PDB)",
        "unit":        "ms",
        "direction":   "lower_is_better",
        "note":        "E2E delay from device to server; does not isolate RAN from core.",
    },
    "jitter_ms": {
        "domain":      "core",
        "3gpp_analog": "Packet Delay Variation (PDV)",
        "unit":        "ms",
        "direction":   "lower_is_better",
        "note":        "Delay variation; affects real-time services (VoNR, streaming).",
    },
    "handover_count": {
        "domain":      "mobility",
        "3gpp_analog": "Handover execution event count",
        "unit":        "count",
        "direction":   "lower_is_better",
        "note":        "High count per session indicates cell-edge or coverage holes.",
    },
    "ping_google_ms": {
        "domain":      "transport",
        "3gpp_analog": "Transport / external Internet RTT (not a 3GPP KPI)",
        "unit":        "ms",
        "direction":   "lower_is_better",
        "note":        "External probe to google.com; high value may indicate N6/backhaul issue.",
    },
    "dropped_connection": {
        "domain":      "retainability",
        "3gpp_analog": "Connection retainability / availability",
        "unit":        "boolean",
        "direction":   "lower_is_better",
        "note":        "Direct availability drop event; strong indicator of service disruption.",
    },
    "network_congestion_level": {
        "domain":      "load",
        "3gpp_analog": "Cell / core load indicator (PRB utilization proxy)",
        "unit":        "categorical (Low / Medium / High)",
        "direction":   "lower_is_better",
        "note":        "Operator-reported categorical label; not a direct 3GPP KPI.",
    },
}

# ============================================================
# Phase 3 -- Centralized KPI threshold table
# ============================================================
# All threshold values are in original physical units.
# Source of truth: internal KPI specification document (v1.0).
# References:
#   RSRP thresholds     : 3GPP TS 38.133 section 10.1.6.1 (adapted)
#   Latency thresholds  : 3GPP TS 22.261 section 7 (5G eMBB service requirements)
#   Jitter (PDV)        : operational heuristic; no standardized 3GPP UE PDV bound
#   Handover count      : heuristic per session observation window
#   Ping to Google      : external RTT probe; not a direct 3GPP KPI
#
# Structure per KPI:
#   normal   : {op, value}          -- condition for NORMAL
#   warning  : {op, low, high}      -- human-readable range label (see note)
#                                      actual evaluation uses normal.value / critical.value
#   critical : {op, value}          -- condition for CRITICAL
#   unit, direction                 -- for _threshold_description() and evaluation branching
#
# Boundary policy (applied in evaluate_kpi_threshold):
#   higher_is_better KPIs  (signal_strength, download_speed, upload_speed):
#     v >= normal.value                        -> NORMAL
#     critical.value <= v < normal.value       -> WARNING  (= boundary belongs to WARNING)
#     v < critical.value                       -> CRITICAL
#     Example: signal = -90 dBm -> NORMAL;  signal = -105 dBm -> WARNING
#
#   lower_is_better KPIs  (latency, jitter, handover, ping):
#     v <= normal.value                        -> NORMAL
#     normal.value < v <= critical.value       -> WARNING  (= boundary belongs to WARNING)
#     v > critical.value                       -> CRITICAL
#     Example: latency = 30 ms -> NORMAL;  latency = 100 ms -> WARNING
KPI_THRESHOLDS: dict[str, dict[str, Any]] = {
    "signal_strength_dbm": {
        # RSRP-like proxy. Doc spec: Normal >= -90 dBm, Warning -105 ~ -90 dBm, Critical < -105 dBm
        # Boundary: -90 dBm = NORMAL;  -105 dBm = WARNING;  -105.001 dBm = CRITICAL
        "normal":    {"op": ">=",    "value": -90.0},
        "warning":   {"op": "range", "low": -105.0, "high": -90.0},
        "critical":  {"op": "<",     "value": -105.0},
        "unit":      "dBm",
        "direction": "higher_is_better",
    },
    "download_speed_mbps": {
        # DL throughput proxy. Doc spec: Normal >= 100 Mbps, Warning 30 ~ 100 Mbps, Critical < 30 Mbps
        "normal":    {"op": ">=",    "value": 100.0},
        "warning":   {"op": "range", "low": 30.0, "high": 100.0},
        "critical":  {"op": "<",     "value": 30.0},
        "unit":      "Mbps",
        "direction": "higher_is_better",
    },
    "upload_speed_mbps": {
        # UL throughput proxy. Doc spec: Normal >= 20 Mbps, Warning 5 ~ 20 Mbps, Critical < 5 Mbps
        "normal":    {"op": ">=",    "value": 20.0},
        "warning":   {"op": "range", "low": 5.0, "high": 20.0},
        "critical":  {"op": "<",     "value": 5.0},
        "unit":      "Mbps",
        "direction": "higher_is_better",
    },
    "latency_ms": {
        # E2E delay / PDB-related observation. Doc spec: Normal <= 30 ms, Warning 30 ~ 100 ms, Critical > 100 ms
        # Boundary: 30 ms = NORMAL;  100 ms = WARNING;  100.001 ms = CRITICAL
        "normal":    {"op": "<=",    "value": 30.0},
        "warning":   {"op": "range", "low": 30.0, "high": 100.0},
        "critical":  {"op": ">",     "value": 100.0},
        "unit":      "ms",
        "direction": "lower_is_better",
    },
    "jitter_ms": {
        # PDV-like metric. Doc spec: Normal <= 10 ms, Warning 10 ~ 30 ms, Critical > 30 ms
        # Boundary: 10 ms = NORMAL;  30 ms = WARNING;  30.001 ms = CRITICAL
        "normal":    {"op": "<=",    "value": 10.0},
        "warning":   {"op": "range", "low": 10.0, "high": 30.0},
        "critical":  {"op": ">",     "value": 30.0},
        "unit":      "ms",
        "direction": "lower_is_better",
    },
    "handover_count": {
        # Heuristic per observation window. Doc spec: Normal 0 ~ 2, Warning 3 ~ 5, Critical > 5
        "normal":    {"op": "<=",    "value": 2},
        "warning":   {"op": "range", "low": 3, "high": 5},
        "critical":  {"op": ">",     "value": 5},
        "unit":      "count",
        "direction": "lower_is_better",
    },
    "ping_google_ms": {
        # External RTT proxy; not a direct 3GPP KPI.
        # Doc spec: Normal <= 50 ms, Warning 50 ~ 150 ms, Critical > 150 ms
        # Boundary: 50 ms = NORMAL;  150 ms = WARNING;  150.001 ms = CRITICAL
        "normal":    {"op": "<=",    "value": 50.0},
        "warning":   {"op": "range", "low": 50.0, "high": 150.0},
        "critical":  {"op": ">",     "value": 150.0},
        "unit":      "ms",
        "direction": "lower_is_better",
    },
    "dropped_connection": {
        # Retainability / availability symptom. Boolean: True = service disruption event.
        # No intermediate warning state; a drop is always CRITICAL.
        "normal":    {"op": "==", "value": False},
        "warning":   {"op": "==", "value": False},   # no intermediate state
        "critical":  {"op": "==", "value": True},
        "unit":      "boolean",
        "direction": "lower_is_better",
    },
    "network_congestion_level": {
        # Categorical congestion indicator (operator-assigned label; PRB utilization proxy).
        # Dataset column is a categorical label, NOT a raw PRB% measurement.
        "normal":    {"op": "==", "value": "Low"},
        "warning":   {"op": "==", "value": "Medium"},
        "critical":  {"op": "==", "value": "High"},
        "unit":      "categorical",
        "direction": "lower_is_better",
    },
}

# Secondary device-side thresholds (NOT primary KPIs).
# Used ONLY as contributing factors in root-cause inference (R5 rule support).
# Battery / temperature alone do NOT directly set overall severity;
# they only strengthen the case for RootCause.DEVICE_SIDE when QoE is also degraded.
#
# Doc spec:
#   battery_level_percent : Normal >= 20%, Warning 10 ~ 20%, Critical < 10%
#   temperature_celsius   : Normal <= 40 °C, Warning 40 ~ 45 °C, Critical > 45 °C
DEVICE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "battery_level_percent": {
        "warning_threshold":  20.0,   # < 20% : at least warning-level device concern (R5 onset)
        "critical_threshold": 10.0,   # < 10% : critical device constraint
        "unit":               "%",
        "note":               "Below warning_threshold, modem may reduce TX power (device-side).",
    },
    "temperature_celsius": {
        "warning_threshold":  40.0,   # > 40 °C : at least warning-level thermal concern (R5 onset)
        "critical_threshold": 45.0,   # > 45 °C : critical thermal throttling risk
        "unit":               "celsius",
        "note":               "Above warning_threshold, thermal throttling may degrade modem throughput.",
    },
}

# ============================================================
# Phase 3 -- Context / secondary indicators
#            (NOT included in primary severity aggregate)
# ============================================================
# These are NOT swept by evaluate_all_kpis() and do NOT feed compute_composite_severity().
# They are used only for:
#   - QoE degradation context reinforcement (R5 device-side support)
#   - Explainability annotations in diagnosis output
#
# video_streaming_quality : subjective QoE user-experience score
#   Normal  : 4 ~ 5  (Good)
#   Warning : 2 ~ 3  (Fair)
#   Critical: 1      (Poor)
#
# connected_duration_min  : session stability proxy
#   TODO: requires session-fragmentation / reconnect-count metadata
#   for robust thresholding. No numeric hard threshold in specification document.
CONTEXT_THRESHOLDS: dict[str, Any] = {
    "video_streaming_quality": {
        "normal_min":  4,   # 4 or 5 = Good (Normal)
        "warning_min": 2,   # 2 or 3 = Fair (Warning)
        "critical":    1,   # 1      = Poor (Critical)
        "note": (
            "Subjective QoE experience score (1-5 scale). "
            "Secondary indicator; does NOT directly contribute to compute_composite_severity(). "
            "Used to strengthen QoE degradation evidence in R5."
        ),
    },
    "connected_duration_min": {
        # TODO: requires session-fragmentation/reconnect-count metadata
        # for robust thresholding. No numeric bound in specification document.
        "note": (
            "Session duration proxy. Hard numeric threshold deliberately omitted; "
            "available as context/metadata only."
        ),
    },
}

# ============================================================
# Enums
# ============================================================
class SeverityLevel(Enum):
    NORMAL   = "normal"
    WARNING  = "warning"
    CRITICAL = "critical"

    def __lt__(self, other: "SeverityLevel") -> bool:
        _order = {SeverityLevel.NORMAL: 0, SeverityLevel.WARNING: 1, SeverityLevel.CRITICAL: 2}
        return _order[self] < _order[other]

    def __le__(self, other: "SeverityLevel") -> bool:
        return self == other or self < other


class RootCause(Enum):
    # Radio layer
    RADIO_COVERAGE      = "radio_coverage_degradation"
    RADIO_RETAINABILITY = "radio_retainability_failure"
    CELL_EDGE_MOBILITY  = "cell_edge_or_mobility_instability"
    # Core / backhaul
    CORE_CONGESTION     = "core_or_backhaul_congestion"
    TRANSPORT_ISSUE     = "transport_or_core_issue"
    # Device
    DEVICE_SIDE         = "device_side_factor"
    # Combined
    MULTI_DOMAIN        = "multi_domain_degradation"
    # Default
    INDETERMINATE       = "indeterminate"


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class BreachedKPI:
    """A single KPI that exceeded its normal threshold."""
    kpi:       str
    value:     float | bool | str
    level:     SeverityLevel
    threshold: str              # human-readable threshold label


@dataclass
class AttentionProxy:
    """
    Explainability proxy -- NOT transformer self-attention.

    Three components:
      feature_importance_target_download :
        Importance of each input feature toward predicting download speed.
        Source: RF impurity importance (if sklearn available) or domain-expert
        rule weights.
      feature_importance_target_drop :
        Same, but toward predicting dropped_connection.
      local_rule_weights :
        For THIS specific record, which breached KPIs contributed most
        to the diagnosis decision (normalized severity weights).
      method :
        "rf_impurity"  -- scikit-learn RandomForest impurity importance
        "rule_based"   -- domain-expert fallback weights
    """
    feature_importance_target_download: dict[str, float] = field(default_factory=dict)
    feature_importance_target_drop:     dict[str, float] = field(default_factory=dict)
    local_rule_weights:                 dict[str, float] = field(default_factory=dict)
    method:                             str = "rule_based"


# ============================================================
# LLM integration dataclass  (used in Phase 8)
# ============================================================
@dataclass
class LLMValidation:
    """
    Result of calling the LLM server for one record and comparing
    its response against the rule-based diagnosis.

    Fields
    ------
    prompt            : exact user prompt sent to the model
    response          : raw text returned by the model
    parsed_severity   : 'severity:' line extracted from the response
    parsed_root_cause : 'root_cause:' line extracted from the response
    severity_match    : True iff parsed_severity == rule severity_level
    root_cause_match  : True iff parsed_root_cause == rule inferred_root_cause
    match_overall     : True iff both fields match
    error             : non-empty if the LLM call itself failed
    """
    prompt:             str
    response:           str
    parsed_severity:    str
    parsed_root_cause:  str
    severity_match:     bool
    root_cause_match:   bool
    match_overall:      bool
    error:              str = ""


@dataclass
class DiagnosisRecord:
    """Full per-record diagnosis output."""
    timestamp:             str
    severity_level:        SeverityLevel
    breached_kpis:         list[BreachedKPI]
    inferred_root_cause:   RootCause
    confidence_rule_basis: str    # which rules fired (plain English)
    recommended_action:    str
    attention_proxy:       AttentionProxy
    raw:                   dict[str, Any]    = field(default_factory=dict)
    llm_validation:        "LLMValidation | None" = None


@dataclass
class SummaryReport:
    """Aggregate statistics across all diagnosed records."""
    total_records:             int
    counts_by_severity:        dict[str, int]
    counts_by_root_cause:      dict[str, int]
    top_features_download:     list[tuple[str, float]]
    top_features_drop:         list[tuple[str, float]]
    threshold_violation_stats: dict[str, dict[str, int]]   # {kpi: {level: count}}


# ============================================================
# Phase 0 -- safe_ops runtime configuration
# ============================================================

def configure_safe_ops_runtime(
    intra_threads: int = 4,
    interop_threads: int = 1,
) -> tuple[int, int]:
    """
    Configure PyTorch CPU thread counts.

    Preference order:
      1. safe_ops.configure_runtime()  -- C++ extension (no GIL release needed)
      2. torch.set_num_threads()       -- Python fallback
      3. Return (-1, -1)               -- no torch, no safe_ops

    Returns
    -------
    (actual_intra_threads, actual_interop_threads)
    """
    if _HAS_SAFE_OPS:
        try:
            result = _safe_ops.configure_runtime(intra_threads, interop_threads)
            return (int(result[0]), int(result[1]))
        except Exception as exc:
            warnings.warn(f"safe_ops.configure_runtime failed: {exc}")
    # Python / torch fallback
    try:
        import torch  # noqa: PLC0415
        if intra_threads > 0:
            torch.set_num_threads(intra_threads)
        return (torch.get_num_threads(), torch.get_num_interop_threads())
    except ImportError:
        warnings.warn("Neither safe_ops nor torch available; thread config skipped.")
        return (-1, -1)


def probe_cuda_safe_ops(
    target_mb:  int = 640,
    reserve_mb: int = 512,
    step_mb:    int = 64,
    min_mb:     int = 256,
) -> dict[str, int | str]:
    """
    Conservative CUDA-memory budget probe.

    Preference order:
      1. safe_ops.probe_cuda_budget()   -- real cudaMalloc-based probe
         (accounts for fragmentation; more conservative than cudaMemGetInfo)
      2. torch.cuda.mem_get_info()      -- simple free/total query (fallback)
      3. Return all-zero dict           -- CUDA not available

    Returns
    -------
    {
      \"free_mb\"        : currently free VRAM in MB,
      \"total_mb\"       : total VRAM in MB,
      \"safe_budget_mb\" : conservative safe allocation budget in MB,
      \"source\"         : \"safe_ops\" | \"torch\" | \"unavailable\"
    }
    """
    if _HAS_SAFE_OPS:
        try:
            result = _safe_ops.probe_cuda_budget(target_mb, reserve_mb, step_mb, min_mb)
            return {
                "free_mb":        int(result[0]),
                "total_mb":       int(result[1]),
                "safe_budget_mb": int(result[2]),
                "source":         "safe_ops",
            }
        except Exception as exc:
            warnings.warn(f"safe_ops.probe_cuda_budget failed: {exc}")
    # torch fallback
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            free_mb  = free_b  // (1024 * 1024)
            total_mb = total_b // (1024 * 1024)
            safe_mb  = max(0, min(free_mb - reserve_mb, target_mb))
            return {
                "free_mb":        free_mb,
                "total_mb":       total_mb,
                "safe_budget_mb": safe_mb,
                "source":         "torch",
            }
    except Exception:
        pass
    return {"free_mb": 0, "total_mb": 0, "safe_budget_mb": 0, "source": "unavailable"}


# ============================================================
# Phase 1 -- Data loading and cleaning
# ============================================================
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV, apply COLUMN_MAP rename, and coerce column types.
    Returns a clean DataFrame with internal snake_case column names.

    Robust to:
      - Columns absent from COLUMN_MAP (kept with original names)
      - Non-numeric values in numeric columns (coerced to NaN)
      - "True"/"False" string representations for boolean columns
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename only columns that actually exist in the DataFrame
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Coerce timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Numeric columns -- coerce to float64 (NaN on parse failure)
    float_cols = [
        "signal_strength_dbm", "download_speed_mbps", "upload_speed_mbps",
        "latency_ms", "jitter_ms", "battery_level_percent", "temperature_celsius",
        "connected_duration_min", "data_usage_mb", "ping_google_ms",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Integer columns
    for col in ("handover_count", "video_streaming_quality"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean columns -- handle string "True"/"False"
    for col in ("vonr_enabled", "dropped_connection"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True if str(v).strip().lower() == "true" else False
            ).astype(bool)

    # Normalize categoricals to title case for consistent matching
    if "network_congestion_level" in df.columns:
        df["network_congestion_level"] = df["network_congestion_level"].str.strip().str.title()
    if "network_type" in df.columns:
        df["network_type"] = df["network_type"].str.strip()

    return df


def check_missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return list of required column names absent from df."""
    return [c for c in required if c not in df.columns]


# ============================================================
# Phase 3 -- Threshold evaluation engine
# ============================================================
def _threshold_description(kpi: str, level: str) -> str:
    """Return a human-readable threshold label for a KPI at a given level."""
    spec = KPI_THRESHOLDS.get(kpi, {})
    t    = spec.get(level, {})
    unit = spec.get("unit", "")
    op   = t.get("op", "")
    if op in (">=", ">", "<=", "<"):
        return f"{op} {t['value']} {unit}".strip()
    if op == "range":
        return f"{t['low']} ~ {t['high']} {unit}".strip()
    if op == "==":
        return f"== {t['value']}".strip()
    return ""


def evaluate_kpi_threshold(kpi: str, value: float | bool | str) -> SeverityLevel:
    """
    Evaluate a single KPI value against KPI_THRESHOLDS.
    Returns SeverityLevel (NORMAL, WARNING, or CRITICAL).

    Branch order:
      1. Boolean (dropped_connection)
      2. Categorical (network_congestion_level)
      3. Numeric -- higher_is_better
      4. Numeric -- lower_is_better

    Boundary policy (mirrors KPI_THRESHOLDS boundary notes):
      higher_is_better: v >= normal.value -> NORMAL; boundary at critical.value = WARNING.
        e.g. signal = -90 dBm -> NORMAL;  signal = -105 dBm -> WARNING.
      lower_is_better : v <= normal.value -> NORMAL; boundary at critical.value = WARNING.
        e.g. latency = 30 ms -> NORMAL;  latency = 100 ms -> WARNING.
    """
    if kpi not in KPI_THRESHOLDS:
        return SeverityLevel.NORMAL

    spec      = KPI_THRESHOLDS[kpi]
    direction = spec.get("direction", "lower_is_better")
    crit_spec = spec["critical"]

    # -- 1. Boolean -----------------------------------------------------------
    if crit_spec.get("op") == "==" and isinstance(value, (bool, np.bool_)):
        return SeverityLevel.CRITICAL if bool(value) == bool(crit_spec["value"]) else SeverityLevel.NORMAL

    # -- 2. Categorical -------------------------------------------------------
    if crit_spec.get("op") == "==" and isinstance(value, str):
        if value == crit_spec["value"]:
            return SeverityLevel.CRITICAL
        if value == spec["warning"]["value"]:
            return SeverityLevel.WARNING
        return SeverityLevel.NORMAL

    # -- 3/4. Numeric ---------------------------------------------------------
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SeverityLevel.NORMAL

    if direction == "higher_is_better":
        if v < crit_spec["value"]:
            return SeverityLevel.CRITICAL
        if v < spec["normal"]["value"]:
            return SeverityLevel.WARNING
        return SeverityLevel.NORMAL
    else:  # lower_is_better
        if v > crit_spec["value"]:
            return SeverityLevel.CRITICAL
        if v > spec["normal"]["value"]:
            return SeverityLevel.WARNING
        return SeverityLevel.NORMAL


def evaluate_all_kpis(row: pd.Series) -> list[BreachedKPI]:
    """
    Sweep all KPIs defined in KPI_THRESHOLDS for a single row.
    Returns a list of BreachedKPI entries with level WARNING or CRITICAL.
    KPIs at NORMAL level are omitted.
    """
    breached: list[BreachedKPI] = []
    for kpi in KPI_THRESHOLDS:
        if kpi not in row.index:
            continue
        val   = row[kpi]
        level = evaluate_kpi_threshold(kpi, val)
        if level != SeverityLevel.NORMAL:
            thresh = _threshold_description(kpi, level.value)
            breached.append(BreachedKPI(kpi=kpi, value=val, level=level, threshold=thresh))
    return breached


# ============================================================
# Phase 3 -- Composite severity scoring
# ============================================================
def compute_composite_severity(breached: list[BreachedKPI]) -> SeverityLevel:
    """
    Reduce per-KPI levels to a single record-level severity.
    Strategy: take the worst level observed across all breached KPIs.
    """
    if any(b.level == SeverityLevel.CRITICAL for b in breached):
        return SeverityLevel.CRITICAL
    if any(b.level == SeverityLevel.WARNING for b in breached):
        return SeverityLevel.WARNING
    return SeverityLevel.NORMAL


# ============================================================
# Phase 4 -- Root-cause inference engine
# ============================================================
# Logic requirements implemented below (from spec):
#
#   R1 : latency AND jitter both elevated:
#        R1a -- signal stable + handover low + congestion HIGH -> core/backhaul congestion
#        R1b -- signal WEAK  + handover HIGH                  -> cell-edge / mobility
#        R1c -- mixed conditions                              -> multi-domain
#   R2 : signal WEAK AND handover HIGH (independent of R1)  -> cell-edge / mobility
#   R3 : dropped + WEAK signal                              -> radio retainability
#   R4 : dropped + GOOD radio + (high ping OR high congestion) -> transport/core
#   R4b: signal CRITICAL (no drop)                         -> radio coverage
#   R5 : (battery LOW OR temperature HIGH) + QoE degradation -> device side (secondary)
#
# Definitions used in helper predicates:
#   "weak signal"     = WARNING or CRITICAL on signal_strength_dbm
#   "critical signal" = CRITICAL on signal_strength_dbm
#   "high jitter"     = WARNING or CRITICAL on jitter_ms
#   "high latency"    = WARNING or CRITICAL on latency_ms
#   "high handover"   = WARNING or CRITICAL on handover_count
#   "high congestion" = network_congestion_level == "High"
#   "high ping"       = WARNING or CRITICAL on ping_google_ms

def infer_root_cause(
    row: pd.Series,
    breached: list[BreachedKPI],
) -> tuple[RootCause, str]:
    """
    Deterministic rule-based root-cause inference.
    Returns (RootCause, confidence_rule_basis).

    Multiple rules may fire simultaneously; the highest-priority cause is
    returned as primary, and all fired rules are included in the basis string.
    """
    breached_map: dict[str, SeverityLevel] = {b.kpi: b.level for b in breached}

    # -- Predicate helpers ----------------------------------------------------
    def _breached(kpi: str) -> bool:
        return kpi in breached_map

    def _is_weak_signal() -> bool:
        return _breached("signal_strength_dbm")

    def _is_signal_critical() -> bool:
        return breached_map.get("signal_strength_dbm") == SeverityLevel.CRITICAL

    def _is_latency_high() -> bool:
        return _breached("latency_ms")

    def _is_jitter_high() -> bool:
        return _breached("jitter_ms")

    def _is_high_handover() -> bool:
        return _breached("handover_count")

    def _is_high_congestion() -> bool:
        return str(row.get("network_congestion_level", "Low")).strip() == "High"

    def _is_dropped() -> bool:
        return bool(row.get("dropped_connection", False))

    def _is_high_ping() -> bool:
        return _breached("ping_google_ms")

    def _is_low_battery() -> bool:
        try:
            # Trigger at warning onset (< 20%); critical onset is < 10%.
            # Used only as R5 device-side contributing factor.
            return float(row.get("battery_level_percent", 100.0)) < DEVICE_THRESHOLDS["battery_level_percent"]["warning_threshold"]
        except (TypeError, ValueError):
            return False

    def _is_high_temp() -> bool:
        try:
            # Trigger at warning onset (> 40 °C); critical onset is > 45 °C.
            # Used only as R5 device-side contributing factor.
            return float(row.get("temperature_celsius", 25.0)) > DEVICE_THRESHOLDS["temperature_celsius"]["warning_threshold"]
        except (TypeError, ValueError):
            return False

    def _has_video_quality_degradation() -> bool:
        """video_streaming_quality below normal_min (4) = at least warning-level QoE impact."""
        try:
            vq = row.get("video_streaming_quality")
            if vq is None or pd.isna(vq):
                return False
            return int(vq) < CONTEXT_THRESHOLDS["video_streaming_quality"]["normal_min"]
        except (TypeError, ValueError):
            return False

    def _has_qoe_degradation() -> bool:
        # Includes video quality as a secondary QoE evidence signal.
        return _breached("download_speed_mbps") or _is_dropped() or _has_video_quality_degradation()

    def _sev_tag(kpi: str) -> str:
        """Return severity tag string for a breached KPI; appended to confidence_rule_basis."""
        level = breached_map.get(kpi)
        if level == SeverityLevel.CRITICAL:
            return "[CRITICAL]"
        if level == SeverityLevel.WARNING:
            return "[WARNING]"
        return ""

    # -- Rule evaluation ------------------------------------------------------
    causes:      list[RootCause] = []
    rules_fired: list[str]       = []

    # R1: latency AND jitter both elevated
    if _is_latency_high() and _is_jitter_high():
        if not _is_weak_signal() and not _is_high_handover() and _is_high_congestion():
            # R1a: radio layer is stable; congestion drives delay/jitter
            causes.append(RootCause.CORE_CONGESTION)
            rules_fired.append(
                f"R1a: latency_high{_sev_tag('latency_ms')}+jitter_high{_sev_tag('jitter_ms')}"
                "+signal_ok+handover_ok+congestion_high"
                " -> core_or_backhaul_congestion"
            )
        elif _is_weak_signal() and _is_high_handover():
            # R1b: radio instability causes HO-induced delay and jitter variation
            causes.append(RootCause.CELL_EDGE_MOBILITY)
            rules_fired.append(
                f"R1b: latency_high{_sev_tag('latency_ms')}+jitter_high{_sev_tag('jitter_ms')}"
                f"+signal_weak{_sev_tag('signal_strength_dbm')}+handover_high{_sev_tag('handover_count')}"
                " -> cell_edge_or_mobility_instability"
            )
        else:
            # R1c: latency+jitter degraded but root-cause evidence is mixed
            causes.append(RootCause.MULTI_DOMAIN)
            rules_fired.append(
                f"R1c: latency_high{_sev_tag('latency_ms')}+jitter_high{_sev_tag('jitter_ms')}"
                ", conditions ambiguous -> multi_domain_degradation"
            )

    # R2: signal weak AND handover high (independent of R1)
    if _is_weak_signal() and _is_high_handover() and RootCause.CELL_EDGE_MOBILITY not in causes:
        causes.append(RootCause.CELL_EDGE_MOBILITY)
        rules_fired.append(
            f"R2: signal_weak{_sev_tag('signal_strength_dbm')}+handover_high{_sev_tag('handover_count')}"
            " -> cell_edge_or_mobility_instability"
        )

    # R3: dropped connection with weak signal -> radio retainability
    if _is_dropped() and _is_weak_signal():
        causes.append(RootCause.RADIO_RETAINABILITY)
        rules_fired.append(
            f"R3: dropped_connection[CRITICAL]+signal_weak{_sev_tag('signal_strength_dbm')}"
            " -> radio_retainability_failure"
        )

    # R4: dropped connection with good radio + high ping or congestion -> transport/core
    if _is_dropped() and not _is_weak_signal() and (_is_high_ping() or _is_high_congestion()):
        causes.append(RootCause.TRANSPORT_ISSUE)
        rules_fired.append(
            f"R4: dropped_connection[CRITICAL]+signal_ok"
            f"+(ping_high{_sev_tag('ping_google_ms')} or congestion_high)"
            " -> transport_or_core_issue"
        )

    # R4b: signal critically low without a drop event -> radio coverage gap
    if _is_signal_critical() and not _is_dropped() and RootCause.CELL_EDGE_MOBILITY not in causes:
        causes.append(RootCause.RADIO_COVERAGE)
        rules_fired.append(
            "R4b: signal_critical[CRITICAL]+no_drop -> radio_coverage_degradation"
        )

    # R5: device conditions with QoE degradation -> secondary device factor
    if (_is_low_battery() or _is_high_temp()) and _has_qoe_degradation():
        causes.append(RootCause.DEVICE_SIDE)
        rules_fired.append(
            "R5: (battery_low or temp_high)+qoe_degraded"
            " -> device_side_factor (secondary; possible contribution)"
        )

    # -- Reduce to primary cause ----------------------------------------------
    if not causes:
        if breached_map:
            return (
                RootCause.INDETERMINATE,
                "KPIs breached but no specific rule matched",
            )
        return RootCause.INDETERMINATE, "no KPI breaches detected"

    if len(causes) == 1:
        return causes[0], "; ".join(rules_fired)

    # Priority order when multiple rules fire.
    # Rationale: service-breaking failures (retainability) outrank slow degradations.
    PRIORITY: list[RootCause] = [
        RootCause.RADIO_RETAINABILITY,
        RootCause.CELL_EDGE_MOBILITY,
        RootCause.CORE_CONGESTION,
        RootCause.TRANSPORT_ISSUE,
        RootCause.MULTI_DOMAIN,
        RootCause.RADIO_COVERAGE,
        RootCause.DEVICE_SIDE,
    ]
    for primary in PRIORITY:
        if primary in causes:
            return primary, "; ".join(rules_fired)

    return RootCause.MULTI_DOMAIN, "; ".join(rules_fired)


# ============================================================
# Phase 5 -- Action recommendation engine
# ============================================================
# Structured lookup: (RootCause, SeverityLevel) -> action string.
# Actions are operational guidance aligned to 3GPP NM / O&M concepts.
# ASSUMPTION: actions are advisory; actual enforcement depends on operator tooling.
ACTION_MAP: dict[RootCause, dict[SeverityLevel, str]] = {
    RootCause.RADIO_RETAINABILITY: {
        SeverityLevel.CRITICAL: (
            "Immediate RAN investigation: verify RACH failure rate, RLF counter, "
            "and RRC re-establishment success ratio. Trigger forced handover or "
            "beam recovery procedure for affected UE."
        ),
        SeverityLevel.WARNING: (
            "Monitor retainability counters (HOSR, RLF rate). Consider beam management "
            "or antenna tilt adjustment to improve coverage at UE location."
        ),
    },
    RootCause.CELL_EDGE_MOBILITY: {
        SeverityLevel.CRITICAL: (
            "UE at cell edge with high handover rate. Tune A3/A5 event thresholds, "
            "reduce TTT, or deploy small cell / repeater in the coverage gap area."
        ),
        SeverityLevel.WARNING: (
            "UE approaching cell boundary. Increase hysteresis margin to reduce "
            "ping-pong handovers; evaluate coverage plot for gap elimination."
        ),
    },
    RootCause.CORE_CONGESTION: {
        SeverityLevel.CRITICAL: (
            "Core/backhaul congestion confirmed: high latency, jitter, and congestion "
            "indicator co-occurring with stable radio. Activate QoS priority elevation "
            "for critical flows; check N6/N9 interface capacity and backhaul link load."
        ),
        SeverityLevel.WARNING: (
            "Backhaul load elevated. Monitor N6 link utilization; consider traffic "
            "offload via edge caching, local breakout (ULCL), or dynamic routing policy."
        ),
    },
    RootCause.TRANSPORT_ISSUE: {
        SeverityLevel.CRITICAL: (
            "Connection dropped despite adequate radio; high ping or congestion "
            "suggests transport/core path failure. Investigate SGW/UPF routing, "
            "N3/N9 GTP-U tunnel integrity, and SCTP association state."
        ),
        SeverityLevel.WARNING: (
            "Transport path congestion with good radio. Check core network load, "
            "routing towards N6, and peering conditions to external internet."
        ),
    },
    RootCause.RADIO_COVERAGE: {
        SeverityLevel.CRITICAL: (
            "RSRP critically low; service continuity at risk. Escalate to RF "
            "optimization: review antenna configuration, coverage gap analysis, "
            "and consider indoor coverage (RIS/repeater) deployment."
        ),
        SeverityLevel.WARNING: (
            "Signal strength marginal. Investigate physical obstruction or coverage "
            "gap; verify handover target cell availability."
        ),
    },
    RootCause.MULTI_DOMAIN: {
        SeverityLevel.CRITICAL: (
            "Simultaneous failures across multiple domains. Perform end-to-end trace "
            "(RAN + transport + core) and correlate alarms; prioritize by SLA impact."
        ),
        SeverityLevel.WARNING: (
            "Cross-domain KPI degradation. Run correlated alarm analysis and check "
            "for shared infrastructure dependency (e.g., common backhaul)."
        ),
    },
    RootCause.DEVICE_SIDE: {
        SeverityLevel.CRITICAL: (
            "Device thermal/battery condition likely limiting modem performance. "
            "Check UE modem power class, enable power-saving offload, review device logs."
        ),
        SeverityLevel.WARNING: (
            "Device-side factor (low battery or high temperature) contributing to "
            "QoE degradation. Consider background-data throttling or UE replacement."
        ),
    },
    RootCause.INDETERMINATE: {
        SeverityLevel.CRITICAL: "Insufficient evidence for root-cause; perform full E2E trace.",
        SeverityLevel.WARNING:  "KPI drift observed; continue monitoring to establish trend.",
        SeverityLevel.NORMAL:   "All KPIs within normal bounds; maintain current configuration.",
    },
}


def recommend_action(root_cause: RootCause, severity: SeverityLevel) -> str:
    """Look up action from ACTION_MAP. Falls back gracefully if a key is absent."""
    cause_map = ACTION_MAP.get(root_cause, ACTION_MAP[RootCause.INDETERMINATE])
    return cause_map.get(
        severity,
        cause_map.get(SeverityLevel.WARNING, "Monitor KPI trends closely."),
    )


# ============================================================
# Phase 6 -- Feature importance / attention proxy
# ============================================================
# "attention_proxy" in this system = EXPLAINABILITY PROXY.
# It is NOT transformer self-attention.
#
# Implementation:
#   - Global level : RandomForest impurity importance (sklearn) per target KPI
#   - Fallback     : domain-expert rule weights (when sklearn unavailable)
#   - Local level  : per-record breached-KPI severity weighting
#
# Features fed to the model are numeric columns + encoded congestion level.
# Normalized copies are created only here (for ML input); raw values are
# retained for all threshold logic.
ORDERED_NUMERIC_FEATURES: list[str] = [
    "signal_strength_dbm",
    "latency_ms",
    "jitter_ms",
    "handover_count",
    "ping_google_ms",
    "battery_level_percent",
    "temperature_celsius",
    "connected_duration_min",
    "upload_speed_mbps",
]

CONGESTION_ENCODING: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build numeric feature matrix for ML. Returns (X, feature_name_list)."""
    feat_cols = [c for c in ORDERED_NUMERIC_FEATURES if c in df.columns]
    X = df[feat_cols].copy()
    if "network_congestion_level" in df.columns:
        X["congestion_encoded"] = (
            df["network_congestion_level"].map(CONGESTION_ENCODING).fillna(1)
        )
        feat_cols = feat_cols + ["congestion_encoded"]
    # Fill NaN with column median for ML stability (only in X copy, not original df)
    X = X.fillna(X.median(numeric_only=True))
    return X, feat_cols


def _rule_based_feature_proxy(feat_cols: list[str]) -> AttentionProxy:
    """
    Domain-expert rule weights as fallback.
    Weights reflect 3GPP domain knowledge about factors driving each target KPI.
    These are NOT learned; treat them as prior engineering estimates.
    """
    # Factors most strongly correlated with download speed (3GPP domain knowledge):
    #   RSRP -> dominant radio-layer factor
    #   Latency/congestion -> capacity / resource contention indicator
    dl_weights_full: dict[str, float] = {
        "signal_strength_dbm":    0.30,
        "congestion_encoded":     0.20,
        "latency_ms":             0.18,
        "ping_google_ms":         0.10,
        "jitter_ms":              0.08,
        "handover_count":         0.07,
        "upload_speed_mbps":      0.04,
        "temperature_celsius":    0.02,
        "battery_level_percent":  0.01,
    }
    # Factors most strongly associated with dropped_connection:
    #   RSRP -> coverage / retainability
    #   Handover count -> mobility instability
    drop_weights_full: dict[str, float] = {
        "signal_strength_dbm":    0.35,
        "handover_count":         0.20,
        "congestion_encoded":     0.15,
        "ping_google_ms":         0.13,
        "latency_ms":             0.09,
        "jitter_ms":              0.04,
        "battery_level_percent":  0.02,
        "temperature_celsius":    0.02,
    }
    fi_dl   = {k: v for k, v in dl_weights_full.items()   if k in feat_cols}
    fi_drop = {k: v for k, v in drop_weights_full.items() if k in feat_cols}
    return AttentionProxy(
        feature_importance_target_download=fi_dl,
        feature_importance_target_drop=fi_drop,
        local_rule_weights={},
        method="rule_based",
    )


def compute_global_feature_importance(df: pd.DataFrame) -> AttentionProxy:
    """
    Compute global feature importance using RandomForest.
    For download_speed_mbps : regression (RandomForestRegressor)
    For dropped_connection   : classification (RandomForestClassifier)

    Falls back to _rule_based_feature_proxy() when:
      - scikit-learn is not installed, OR
      - fewer than 30 rows are available (insufficient for reliable RF)

    IMPORTANT: This function uses a copy of features inside the RF model.
    Original physical-unit values are NOT modified in the DataFrame.
    """
    X, feat_cols = _build_feature_matrix(df)

    if not SKLEARN_AVAILABLE or len(df) < 30:
        return _rule_based_feature_proxy(feat_cols)

    fi_download: dict[str, float] = {}
    fi_drop:     dict[str, float] = {}
    method = "rf_impurity"

    # -- Target 1: download_speed_mbps (regression) --------------------------
    if "download_speed_mbps" in df.columns:
        y_dl = df["download_speed_mbps"].fillna(df["download_speed_mbps"].median())
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y_dl)
            fi_download = dict(
                sorted(zip(feat_cols, rf.feature_importances_), key=lambda t: t[1], reverse=True)
            )
        except Exception as exc:
            warnings.warn(f"RF regressor failed ({exc}); using rule-based weights for download.")
            fi_download = _rule_based_feature_proxy(feat_cols).feature_importance_target_download

    # -- Target 2: dropped_connection (classification) -----------------------
    if "dropped_connection" in df.columns:
        y_drop = df["dropped_connection"].astype(int)
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y_drop)
            fi_drop = dict(
                sorted(zip(feat_cols, rf.feature_importances_), key=lambda t: t[1], reverse=True)
            )
        except Exception as exc:
            warnings.warn(f"RF classifier failed ({exc}); using rule-based weights for drop.")
            fi_drop = _rule_based_feature_proxy(feat_cols).feature_importance_target_drop

    return AttentionProxy(
        feature_importance_target_download=fi_download,
        feature_importance_target_drop=fi_drop,
        local_rule_weights={},    # populated at record level
        method=method,
    )


def compute_local_rule_weights(breached: list[BreachedKPI]) -> dict[str, float]:
    """
    Per-record explainability: weight each breached KPI by its severity
    and domain relevance.

    CRITICAL breached KPI -> base weight 1.0
    WARNING  breached KPI -> base weight 0.5
    Device-domain KPIs are down-weighted (secondary factors).

    Result is normalized to sum=1.0.
    This provides a per-record "which KPIs drove this diagnosis" signal.
    """
    SEVERITY_MULT: dict[SeverityLevel, float] = {
        SeverityLevel.CRITICAL: 1.0,
        SeverityLevel.WARNING:  0.5,
    }
    weights: dict[str, float] = {}
    for b in breached:
        base          = SEVERITY_MULT.get(b.level, 0.0)
        domain_factor = 0.7 if KPI_META.get(b.kpi, {}).get("domain") == "device" else 1.0
        weights[b.kpi] = base * domain_factor

    total = sum(weights.values())
    if total > 0.0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


# ============================================================
# Phase 6 -- Full per-record diagnosis
# ============================================================
def diagnose_record(row: pd.Series, global_proxy: AttentionProxy) -> DiagnosisRecord:
    """
    Run the complete 5-phase diagnosis for one DataFrame row.
    Returns a DiagnosisRecord.
    """
    ts       = str(row.get("timestamp", ""))
    breached = evaluate_all_kpis(row)                           # Phase 3
    severity = compute_composite_severity(breached)             # Phase 3
    root_cause, confidence = infer_root_cause(row, breached)    # Phase 4
    action   = recommend_action(root_cause, severity)           # Phase 5
    local_w  = compute_local_rule_weights(breached)             # Phase 6

    attention = AttentionProxy(
        feature_importance_target_download=global_proxy.feature_importance_target_download,
        feature_importance_target_drop=global_proxy.feature_importance_target_drop,
        local_rule_weights=local_w,
        method=global_proxy.method,
    )
    # Serialize numpy scalars to Python native types for JSON compatibility
    raw = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}

    return DiagnosisRecord(
        timestamp=ts,
        severity_level=severity,
        breached_kpis=breached,
        inferred_root_cause=root_cause,
        confidence_rule_basis=confidence,
        recommended_action=action,
        attention_proxy=attention,
        raw=raw,
    )


# ============================================================
# Phase 7 -- Summary report
# ============================================================
def build_summary_report(
    records: list[DiagnosisRecord],
    global_proxy: AttentionProxy,
    total_rows: int,
) -> SummaryReport:
    """Aggregate counts and statistics across all diagnosed records."""
    severity_counts: dict[str, int] = {s.value: 0 for s in SeverityLevel}
    cause_counts:    dict[str, int] = {}
    kpi_stats:       dict[str, dict[str, int]] = {}

    for rec in records:
        severity_counts[rec.severity_level.value] += 1
        key = rec.inferred_root_cause.value
        cause_counts[key] = cause_counts.get(key, 0) + 1
        for b in rec.breached_kpis:
            if b.kpi not in kpi_stats:
                kpi_stats[b.kpi] = {s.value: 0 for s in SeverityLevel}
            kpi_stats[b.kpi][b.level.value] += 1

    top_dl   = sorted(global_proxy.feature_importance_target_download.items(), key=lambda t: t[1], reverse=True)[:5]
    top_drop = sorted(global_proxy.feature_importance_target_drop.items(),     key=lambda t: t[1], reverse=True)[:5]

    return SummaryReport(
        total_records=total_rows,
        counts_by_severity=severity_counts,
        counts_by_root_cause=cause_counts,
        top_features_download=top_dl,
        top_features_drop=top_drop,
        threshold_violation_stats=kpi_stats,
    )


# ============================================================
# Phase 7 -- Export helpers (I/O separated from pure logic)
# ============================================================
def _serialize_record(rec: DiagnosisRecord) -> dict:
    """Convert DiagnosisRecord to a JSON-serializable dict."""
    breached_list = [
        {
            "kpi":       b.kpi,
            "value":     bool(b.value) if isinstance(b.value, (bool, np.bool_)) else b.value,
            "level":     b.level.value,
            "threshold": b.threshold,
        }
        for b in rec.breached_kpis
    ]
    d: dict = {
        "timestamp":             rec.timestamp,
        "severity_level":        rec.severity_level.value,
        "breached_kpis":         breached_list,
        "inferred_root_cause":   rec.inferred_root_cause.value,
        "confidence_rule_basis": rec.confidence_rule_basis,
        "recommended_action":    rec.recommended_action,
        "attention_proxy": {
            "feature_importance_target_download": rec.attention_proxy.feature_importance_target_download,
            "feature_importance_target_drop":     rec.attention_proxy.feature_importance_target_drop,
            "local_rule_weights":                 rec.attention_proxy.local_rule_weights,
            "method":                             rec.attention_proxy.method,
        },
    }
    # Always embed llm_prompt so the test shell script can call the server
    # independently (even when --llm was not passed).
    try:
        row_proxy = pd.Series(rec.raw)
        d["llm_prompt"] = build_llm_user_prompt(rec, row_proxy)
    except Exception:
        d["llm_prompt"] = ""

    if rec.llm_validation is not None:
        lv = rec.llm_validation
        d["llm_response"] = lv.response
        d["llm_validation"] = {
            "parsed_severity":    lv.parsed_severity,
            "parsed_root_cause":  lv.parsed_root_cause,
            "severity_match":     lv.severity_match,
            "root_cause_match":   lv.root_cause_match,
            "match_overall":      lv.match_overall,
            "error":              lv.error,
        }
    return d


def export_diagnosis_jsonl(records: list[DiagnosisRecord], out_path: Path) -> None:
    """Write one JSON line per record to a JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_serialize_record(rec), ensure_ascii=False) + "\n")
    print(f"[export] diagnosis -> {out_path}  ({len(records)} records)")


def export_summary_json(report: SummaryReport, out_path: Path) -> None:
    """Write aggregated summary statistics to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_records":             report.total_records,
        "counts_by_severity":        report.counts_by_severity,
        "counts_by_root_cause":      report.counts_by_root_cause,
        "top_features_download":     [{"feature": k, "importance": round(v, 4)} for k, v in report.top_features_download],
        "top_features_drop":         [{"feature": k, "importance": round(v, 4)} for k, v in report.top_features_drop],
        "threshold_violation_stats": report.threshold_violation_stats,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[export] summary  -> {out_path}")


# ============================================================
# Phase 8 -- LLM integration: prompt builder, client, validator
# ============================================================

# System prompt instructs the model to emit a parseable structured format.
# Root-cause labels MUST match RootCause enum .value strings exactly.
LLM_SYSTEM_PROMPT: str = (
    "You are a 5G network QoS/QoE diagnostics assistant. "
    "Given network KPI measurements and a pre-computed rule analysis, "
    "respond EXACTLY in this format (no extra text):\n\n"
    "severity: <normal|warning|critical>\n"
    "root_cause: <radio_coverage_degradation|radio_retainability_failure|"
    "cell_edge_or_mobility_instability|core_or_backhaul_congestion|"
    "transport_or_core_issue|device_side_factor|multi_domain_degradation|indeterminate>\n"
    "action: <one concise action recommendation>"
)


def build_llm_user_prompt(rec: DiagnosisRecord, row: pd.Series) -> str:
    """
    Build a structured user prompt for Llama-1B from a DiagnosisRecord + raw row.

    Layout:
      - KPI measurement table  (raw physical-unit values)
      - Breached KPI list      (from rule engine)
      - Rules fired            (confidence_rule_basis string)
      - Final request sentence
    """
    kpi_display: list[tuple[str, str, str]] = [
        ("signal_strength_dbm",     "Signal Strength (RSRP-proxy)",  "dBm"),
        ("download_speed_mbps",     "Download Speed",                "Mbps"),
        ("upload_speed_mbps",       "Upload Speed",                  "Mbps"),
        ("latency_ms",              "E2E Latency (PDB-proxy)",       "ms"),
        ("jitter_ms",               "Jitter (PDV-proxy)",            "ms"),
        ("handover_count",          "Handover Count",                ""),
        ("network_congestion_level","Congestion Level",              ""),
        ("ping_google_ms",          "Internet Ping (RTT)",           "ms"),
        ("dropped_connection",      "Dropped Connection",            ""),
        ("battery_level_percent",   "Battery Level",                 "%"),
        ("temperature_celsius",     "Temperature",                   "°C"),
    ]
    kpi_lines: list[str] = []
    for col, label, unit in kpi_display:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            kpi_lines.append(f"  {label}: {val}{' ' + unit if unit else ''}")

    breached_lines: list[str] = (
        [f"  - {b.kpi}: {b.value} [{b.level.value.upper()}]" for b in rec.breached_kpis]
        if rec.breached_kpis else ["  (none)"]
    )

    return (
        f"Network measurement (Timestamp: {rec.timestamp}):\n"
        + "\n".join(kpi_lines)
        + "\n\nBreached KPIs:\n"
        + "\n".join(breached_lines)
        + f"\n\nRule engine (rules fired): {rec.confidence_rule_basis}\n"
        + "\nDiagnose the network condition."
    )


def call_llm_chat(
    system_prompt:  str,
    user_prompt:    str,
    host:           str   = LLM_DEFAULT_HOST,
    port:           int   = LLM_DEFAULT_PORT,
    max_new_tokens: int   = LLM_DEFAULT_MAX_NEW_TOKENS,
    timeout:        float = 30.0,
) -> str:
    """
    POST to /v1/chat/completions on the inference server.
    Returns the generated text string, or '__LLM_ERROR__: <reason>' on failure.

    Uses only stdlib urllib (no requests/httpx dependency).
    """
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "stop": ["\n\n"],
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
    except urllib.error.URLError as exc:
        return f"__LLM_ERROR__: {exc.reason}"
    except Exception as exc:
        return f"__LLM_ERROR__: {exc}"


def parse_llm_response(llm_text: str) -> tuple[str, str]:
    """
    Parse structured LLM output for 'severity:' and 'root_cause:' fields.
    Returns ("", "") if the response does not match the expected format.
    Normalises hyphens -> underscores in root_cause to tolerate LLM drift.
    """
    severity   = ""
    root_cause = ""
    for line in llm_text.strip().splitlines():
        ls = line.strip().lower()
        m = re.match(r"severity\s*:\s*(\S+)", ls)
        if m:
            severity = m.group(1).rstrip(".,;")
        m = re.match(r"root_cause\s*:\s*(.+)", ls)
        if m:
            root_cause = m.group(1).strip().rstrip(".,;").replace("-", "_")
    return severity, root_cause


def validate_llm_vs_rule(
    prompt:   str,
    llm_text: str,
    rec:      DiagnosisRecord,
) -> LLMValidation:
    """
    Compare parsed LLM output against the rule-based DiagnosisRecord.
    Returns an LLMValidation with match flags set.
    """
    if llm_text.startswith("__LLM_ERROR__"):
        return LLMValidation(
            prompt=prompt, response=llm_text,
            parsed_severity="", parsed_root_cause="",
            severity_match=False, root_cause_match=False,
            match_overall=False, error=llm_text,
        )
    parsed_sev, parsed_rc = parse_llm_response(llm_text)
    sev_match = parsed_sev == rec.severity_level.value
    rc_match  = parsed_rc  == rec.inferred_root_cause.value
    return LLMValidation(
        prompt=prompt,
        response=llm_text,
        parsed_severity=parsed_sev,
        parsed_root_cause=parsed_rc,
        severity_match=sev_match,
        root_cause_match=rc_match,
        match_overall=sev_match and rc_match,
    )




# ============================================================
# Example test rows
# ============================================================
# Three hand-crafted rows cover distinct root-cause scenarios.
# Used for quick local validation via run_example_tests().
# All expected results are verified against doc-spec KPI thresholds (v1.0).
#
# Row A -- Radio retainability failure (dropped + critically weak signal)
#   Expected: severity=CRITICAL, root_cause=radio_retainability_failure
#   Trigger : R3 (dropped_connection=True + signal_strength < -105 dBm [CRITICAL])
#
# Row B -- Transport/core issue (dropped + good radio + warning-level ping)
#   Expected: severity=CRITICAL, root_cause=transport_or_core_issue
#   Trigger : R4 (dropped=True + signal_ok [-73.3 >= -90] + ping_high [87.5 in 50~150])
#   NOTE    : dropped_connection must be True for R4 to fire; ping alone is insufficient.
#
# Row C -- Warning-level external RTT, all other KPIs normal
#   Expected: severity=WARNING, root_cause=indeterminate
#   Reason  : ping=75.5 ms in [50, 150] = WARNING; no R1/R2/R3/R4/R4b/R5 fires.
#
# How to cite these rows in a research report:
#   "Table X illustrates three representative records produced by the rule engine.
#    Row A demonstrates radio retainability failure: signal_strength -108.6 dBm
#    (CRITICAL per doc spec < -105 dBm), concurrent dropped_connection=True,
#    triggering rule R3 -> radio_retainability_failure.
#    Row B demonstrates a transport/core issue: signal -73.3 dBm (NORMAL >= -90 dBm),
#    ping_google_ms 87.5 ms (WARNING, 50 ~ 150 ms), dropped_connection=True,
#    triggering rule R4 -> transport_or_core_issue.
#    Row C demonstrates isolated external RTT degradation: ping 75.5 ms (WARNING,
#    50 ~ 150 ms), all other KPIs within normal bounds; no causal rule fires,
#    resulting in INDETERMINATE root cause."
EXAMPLE_TEST_ROWS: list[dict[str, Any]] = [
    {
        # Row A -- Radio retainability failure
        # signal=-108.6 < -105 dBm -> CRITICAL; dropped=True -> CRITICAL
        # -> R3 fires -> radio_retainability_failure
        "timestamp":               "2025-05-28 06:59:51",
        "location":                "San Francisco",
        "signal_strength_dbm":     -108.6,   # CRITICAL (< -105 dBm per doc spec)
        "download_speed_mbps":     714.94,
        "upload_speed_mbps":       60.41,
        "latency_ms":              10.0,
        "jitter_ms":               4.09,
        "network_type":            "5G NSA",
        "device_model":            "iPhone 14",
        "carrier":                 "AT&T",
        "band":                    "n78",
        "battery_level_percent":   99.0,
        "temperature_celsius":     35.5,
        "connected_duration_min":  14,
        "handover_count":          1,
        "data_usage_mb":           97.4,
        "video_streaming_quality": 4,
        "vonr_enabled":            False,
        "network_congestion_level": "High",
        "ping_google_ms":          27.9,      # NORMAL (<= 50 ms)
        "dropped_connection":      True,      # CRITICAL
    },
    {
        # Row B -- Transport/core issue (dropped + good radio + warning-level ping)
        # signal=-73.3 >= -90 dBm -> NORMAL (signal_ok condition for R4)
        # ping=87.5 in [50, 150] -> WARNING (triggers _is_high_ping())
        # dropped=True -> CRITICAL; R4 fires -> transport_or_core_issue
        "timestamp":               "2025-05-28 06:29:51",
        "location":                "New York",
        "signal_strength_dbm":     -73.3,     # NORMAL (>= -90 dBm)
        "download_speed_mbps":     208.56,
        "upload_speed_mbps":       68.59,
        "latency_ms":              12.2,
        "jitter_ms":               4.94,
        "network_type":            "4G",
        "device_model":            "Pixel 7",
        "carrier":                 "T-Mobile",
        "band":                    "n41",
        "battery_level_percent":   25.0,
        "temperature_celsius":     39.3,
        "connected_duration_min":  48,
        "handover_count":          0,
        "data_usage_mb":           128.87,
        "video_streaming_quality": 4,
        "vonr_enabled":            False,
        "network_congestion_level": "High",
        "ping_google_ms":          87.5,      # WARNING (50 ~ 150 ms)
        "dropped_connection":      True,      # CRITICAL -- required for R4 to fire
    },
    {
        # Row C -- Isolated external RTT degradation; all other KPIs normal
        # ping=75.5 in [50, 150] -> WARNING; no other KPI breached
        # No R1/R2/R3/R4/R4b/R5 fires -> INDETERMINATE
        "timestamp":               "2025-05-28 06:39:51",
        "location":                "Chennai",
        "signal_strength_dbm":     -67.5,     # NORMAL (>= -90 dBm)
        "download_speed_mbps":     796.34,
        "upload_speed_mbps":       136.33,
        "latency_ms":              19.9,      # NORMAL (<= 30 ms)
        "jitter_ms":               1.22,      # NORMAL (<= 10 ms)
        "network_type":            "5G NSA",
        "device_model":            "iPhone 14",
        "carrier":                 "Airtel",
        "band":                    "n78",
        "battery_level_percent":   77.0,
        "temperature_celsius":     36.1,
        "connected_duration_min":  45,
        "handover_count":          2,
        "data_usage_mb":           179.15,
        "video_streaming_quality": 5,
        "vonr_enabled":            False,
        "network_congestion_level": "Low",
        "ping_google_ms":          75.5,      # WARNING (50 ~ 150 ms)
        "dropped_connection":      False,
    },
]


def run_example_tests() -> None:
    """
    Validate the diagnosis pipeline on EXAMPLE_TEST_ROWS and print results.
    Intended for development-time sanity checking, not production evaluation.
    """
    print("\n" + "=" * 72)
    print("EXAMPLE TEST ROWS -- PER-RECORD DIAGNOSIS")
    print("=" * 72)
    proxy = _rule_based_feature_proxy(ORDERED_NUMERIC_FEATURES + ["congestion_encoded"])

    for i, row_dict in enumerate(EXAMPLE_TEST_ROWS):
        row = pd.Series(row_dict)
        rec = diagnose_record(row, proxy)
        print(f"\n-- Row {i + 1}: {row_dict.get('location', '')} | {row_dict.get('timestamp', '')} --")
        print(f"   severity_level        : {rec.severity_level.value}")
        print(f"   inferred_root_cause   : {rec.inferred_root_cause.value}")
        print(f"   confidence_rule_basis : {rec.confidence_rule_basis}")
        print(f"   recommended_action    :")
        for line in rec.recommended_action.split(". "):
            print(f"     {line}.")
        if rec.breached_kpis:
            print("   breached_kpis         :")
            for b in rec.breached_kpis:
                print(f"     * {b.kpi}: {b.value}  [{b.level.value}]  threshold {b.threshold}")
        print(f"   local_rule_weights    : {rec.attention_proxy.local_rule_weights}")

    print("\n" + "=" * 72)


# ============================================================
# main()
# ============================================================
def main() -> None:
    """
    Full pipeline entry point.

    Phase 0 : safe_ops C++ runtime config (CPU threads + CUDA budget probe)
    Phase 1 : Load & clean CSV; apply MAX_SAMPLES / TEST_SAMPLES cap
    Phase 2 : Verify KPI column availability
    Phase 3 : Evaluate KPI thresholds per row
    Phase 4 : Infer root cause per row
    Phase 5 : Recommend action per row
    Phase 6 : Compute global feature importance (RF or rule-based proxy);
              attach per-record local_rule_weights
    Phase 7 : Export JSONL diagnosis + summary JSON; run example tests
    Phase 8 : (--llm) Build structured Llama-1B prompts, call server,
              validate LLM response vs rule engine output
    """
    parser = argparse.ArgumentParser(
        description="5G Network Traffic QoS/QoE Diagnosis Engine"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help=f"Maximum records to process (default: {MAX_SAMPLES})",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"Test mode: process only {TEST_SAMPLES} records",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM inference + rule-vs-LLM validation (Phase 8)",
    )
    parser.add_argument(
        "--llm-host", type=str, default=LLM_DEFAULT_HOST,
        help=f"LLM server host (default: {LLM_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--llm-port", type=int, default=LLM_DEFAULT_PORT,
        help=f"LLM server port (default: {LLM_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--llm-max-tokens", type=int, default=LLM_DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens for LLM response (default: {LLM_DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--no-example-tests", action="store_true",
        help="Skip hand-crafted example test row validation",
    )
    args = parser.parse_args()

    # Effective cap: --test overrides --max; both override the default MAX_SAMPLES
    effective_max: int = (
        TEST_SAMPLES if args.test
        else (args.max if args.max is not None else MAX_SAMPLES)
    )

    print("[pipeline] 5G Network Traffic QoS/QoE Diagnosis Engine")
    print(f"[pipeline] CSV        : {RAW_CSV}")
    print(f"[pipeline] output_dir : {OUTPUT_DIR}")
    print(f"[pipeline] max_records: {effective_max}")
    if args.llm:
        print(f"[pipeline] LLM mode   : enabled  ({args.llm_host}:{args.llm_port})")

    # -- Phase 0 : safe_ops runtime configuration ---------------------------
    intra, interop = configure_safe_ops_runtime(
        intra_threads   = int(os.getenv("CPU_THREADS",        "4")),
        interop_threads = int(os.getenv("CPU_INTEROP_THREADS","1")),
    )
    src = "safe_ops" if _HAS_SAFE_OPS else "torch/fallback"
    print(f"[phase-0] CPU threads ({src}): intra={intra}, interop={interop}")

    cuda_info = probe_cuda_safe_ops()
    if cuda_info["total_mb"]:
        print(
            f"[phase-0] CUDA memory  : "
            f"{cuda_info['free_mb']} MB free / {cuda_info['total_mb']} MB total"
            f"  safe_budget={cuda_info['safe_budget_mb']} MB"
            f"  source={cuda_info['source']}"
        )
    else:
        print(f"[phase-0] CUDA memory  : unavailable (source={cuda_info['source']})")

    # -- Phase 1 : data loading + cap ---------------------------------------
    df = load_and_clean(RAW_CSV)
    total_loaded = len(df)
    if effective_max < total_loaded:
        df = df.head(effective_max).copy()
        print(
            f"[phase-1] loaded {total_loaded} rows -> using {len(df)}"
            f"  (limit={effective_max})"
        )
    else:
        print(f"[phase-1] loaded {len(df)} rows, {len(df.columns)} columns")

    missing = check_missing_columns(df, list(KPI_THRESHOLDS.keys()))
    if missing:
        print(f"[phase-1] WARNING: missing KPI columns -> {missing}")

    # -- Phase 6 : global feature importance (fit once) ---------------------
    print("[phase-6] computing global feature importance ...")
    global_proxy = compute_global_feature_importance(df)
    print(f"[phase-6] method: {global_proxy.method}")
    if global_proxy.feature_importance_target_download:
        top3_dl = list(global_proxy.feature_importance_target_download.items())[:3]
        print(f"[phase-6] top-3 -> download_speed      : {top3_dl}")
    if global_proxy.feature_importance_target_drop:
        top3_dr = list(global_proxy.feature_importance_target_drop.items())[:3]
        print(f"[phase-6] top-3 -> dropped_connection  : {top3_dr}")

    # -- Phases 3-5 : per-row diagnosis -------------------------------------
    rows_list = list(df.iterrows())
    records: list[DiagnosisRecord] = []
    for _, row in rows_list:
        records.append(diagnose_record(row, global_proxy))
    print(f"[pipeline] diagnosed {len(records)} records")

    # -- Phase 8 : LLM inference + validation (optional) -------------------
    if args.llm:
        print(f"[phase-8] LLM validation — {len(records)} records ...")
        llm_ok      = 0
        llm_err     = 0
        sev_match_n = 0
        rc_match_n  = 0
        for i, (rec, (_, row)) in enumerate(zip(records, rows_list)):
            prompt   = build_llm_user_prompt(rec, row)
            response = call_llm_chat(
                system_prompt   = LLM_SYSTEM_PROMPT,
                user_prompt     = prompt,
                host            = args.llm_host,
                port            = args.llm_port,
                max_new_tokens  = args.llm_max_tokens,
            )
            validation         = validate_llm_vs_rule(prompt, response, rec)
            rec.llm_validation = validation
            if validation.error:
                llm_err += 1
                status = f"ERROR: {validation.error[:60]}"
            else:
                llm_ok      += 1
                sev_match_n += int(validation.severity_match)
                rc_match_n  += int(validation.root_cause_match)
                status = (
                    f"sev={'OK' if validation.severity_match else 'NO':2s}"
                    f"  rc={'OK' if validation.root_cause_match else 'NO':2s}"
                    f"  overall={'MATCH' if validation.match_overall else 'MISMATCH'}"
                )
            print(f"[phase-8] [{i+1:3d}/{len(records)}] {status}")

        if llm_ok > 0:
            print(
                f"[phase-8] severity  accuracy : "
                f"{sev_match_n}/{llm_ok} ({100.*sev_match_n/llm_ok:.1f}%)"
            )
            print(
                f"[phase-8] root_cause accuracy: "
                f"{rc_match_n}/{llm_ok}  ({100.*rc_match_n/llm_ok:.1f}%)"
            )
        if llm_err:
            print(f"[phase-8] LLM call errors : {llm_err}")

    # -- Phase 7 : export ---------------------------------------------------
    report = build_summary_report(records, global_proxy, len(df))
    print("\n[summary] counts_by_severity    :", report.counts_by_severity)
    print("[summary] counts_by_root_cause  :", report.counts_by_root_cause)
    print(
        "[summary] top features -> download :",
        [(f, round(v, 4)) for f, v in report.top_features_download],
    )
    print(
        "[summary] top features -> drop     :",
        [(f, round(v, 4)) for f, v in report.top_features_drop],
    )

    export_diagnosis_jsonl(records, OUTPUT_DIR / "network_traffic_diagnosis.jsonl")
    export_summary_json(report, OUTPUT_DIR / "network_traffic_summary.json")

    if not args.no_example_tests:
        run_example_tests()


if __name__ == "__main__":
    main()

