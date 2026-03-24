"""
SLM Inference Server
====================
Jetson / 임베디드 환경용 소형 언어모델(SLM) 추론 서버.
FastAPI 기반으로 OpenAI-compatible API 를 제공한다.

구조:
  1) Jetson FSDP 패치        — Jetson NGC PyTorch의 FSDP 미지원 문제 우회
  2) AppConfig               — 환경변수에서 설정 로드
  3) MemoryManager            — GPU/CPU 메모리 정리 및 OOM 감지
  4) DtypeResolver            — dtype 문자열 ↔ torch.dtype 변환
  5) ModelLoader              — 토크나이저·모델 로드 (4bit / CUDA / CPU 폴백)
  6) InferenceEngine          — 프롬프트 구성 · 토큰화 · 텍스트 생성
  7) API 스키마 (Pydantic)    — 요청/응답 데이터 모델
  8) FastAPI 앱 + 라우트      — REST 엔드포인트
  9) CLI 엔트리포인트         — uvicorn 실행
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# C++ 확장 (safe_ops) — 빌드되어 있으면 사용, 아니면 Python 폴백
try:
    import safe_ops as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ═══════════════════════════════════════════════════════════════
# 1. Jetson FSDP 패치
#    Jetson NGC PyTorch 빌드에 torch._C._distributed_c10d 가 없어
#    transformers.generate() → is_fsdp_managed_module() 호출 시
#    ModuleNotFoundError 발생. 두 곳 모두 패치가 필요하다.
# ═══════════════════════════════════════════════════════════════
_no_fsdp = lambda _model: False  # noqa: E731

try:
    import transformers.integrations.fsdp as _fsdp_int
    _fsdp_int.is_fsdp_managed_module = _no_fsdp
except Exception:
    pass
try:
    import transformers.generation.utils as _gen_utils
    if hasattr(_gen_utils, "is_fsdp_managed_module"):
        _gen_utils.is_fsdp_managed_module = _no_fsdp
except Exception:
    pass

# BitsAndBytes 4-bit 양자화 사용 가능 여부
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB_CONFIG = True
except Exception:
    BitsAndBytesConfig = None
    HAS_BNB_CONFIG = False


# ═══════════════════════════════════════════════════════════════
# 2. AppConfig — 환경변수 기반 설정
# ═══════════════════════════════════════════════════════════════
class AppConfig:
    """환경변수(.env)에서 모든 설정을 읽어 하나의 객체에 보관한다."""

    def __init__(self) -> None:
        # ── 기본 서비스 설정 ──
        self.app_title: str = os.getenv("APP_TITLE", "slm-inference-service")
        self.model_id: str = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
        self.model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models")

        # ── 디바이스 / dtype ──
        self.has_cuda: bool = torch.cuda.is_available()
        self.requested_device: str = os.getenv("DEVICE", "auto").lower()
        self.device: str = self._resolve_device()
        self.dtype_name: str = os.getenv(
            "DTYPE", "float16" if self.device == "cuda" else "float32"
        ).lower()

        # ── 토큰 한도 ──
        self.max_input_tokens: int = self._require_int("MAX_INPUT_TOKENS")
        self.max_new_tokens: int = self._require_int("MAX_NEW_TOKENS")

        # ── 생성 파라미터 ──
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
        self.top_p: float = float(os.getenv("TOP_P", "0.9"))
        self.top_k: int = int(os.getenv("TOP_K", "40"))
        self.repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.05"))
        self.do_sample_default: str = os.getenv("DO_SAMPLE_DEFAULT", "auto").lower()

        # ── 4-bit 양자화 ──
        self.load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "0") == "1"
        self.bnb_4bit_compute_dtype: str = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16").lower()
        self.bnb_4bit_quant_type: str = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4").lower()
        self.bnb_4bit_use_double_quant: bool = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "0") == "1"
        self.gpu_offload_enabled: bool = os.getenv("GPU_OFFLOAD_ENABLED", "1") == "1"
        self.gpu_fixed_split_enabled: bool = os.getenv("GPU_FIXED_SPLIT_ENABLED", "0") == "1"
        self.gpu_target_memory_mb: int = int(os.getenv("GPU_TARGET_MEMORY_MB", "640"))
        self.gpu_memory_reserve_mb: int = int(os.getenv("GPU_MEMORY_RESERVE_MB", "512"))
        self.gpu_offload_buffers: bool = os.getenv("GPU_OFFLOAD_BUFFERS", "1") == "1"
        self.gpu_probe_step_mb: int = int(os.getenv("GPU_PROBE_STEP_MB", "64"))
        self.gpu_probe_min_mb: int = int(os.getenv("GPU_PROBE_MIN_MB", "256"))

        # ── 워밍업 / 메모리 관리 ──
        self.enable_warmup: bool = os.getenv("ENABLE_WARMUP", "1") == "1"
        self.warmup_max_new_tokens: int = int(os.getenv("WARMUP_MAX_NEW_TOKENS", "8"))
        self.enable_torch_compile: bool = os.getenv("ENABLE_TORCH_COMPILE", "0") == "1"
        self.empty_cache_on_oom: bool = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"
        self.cpu_single_load_margin_mb: int = int(os.getenv("CPU_SINGLE_LOAD_MARGIN_MB", "128"))
        self.cuda_retry_margin_mb: int = int(os.getenv("CUDA_RETRY_MARGIN_MB", "384"))
        self.cpu_threads: int = int(os.getenv("CPU_THREADS", str(max(1, os.cpu_count() or 1))))
        self.cpu_interop_threads: int = int(os.getenv("CPU_INTEROP_THREADS", "1"))
        self.gc_interval: int = int(os.getenv("GC_INTERVAL", "0"))
        self.cpu_force_greedy: bool = os.getenv("CPU_FORCE_GREEDY", "1") == "1"
        self.cpu_force_greedy_max_temp: float = float(os.getenv("CPU_FORCE_GREEDY_MAX_TEMP", "0.2"))

        # ── 시스템 프롬프트 ──
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT",
            "You are a domain assistant for marine DX physical AI and network systems.",
        )

        # ── 모델 경로 결정 (로컬 캐시 우선) ──
        local_path = os.path.join(
            self.model_cache_dir, self.model_id.replace("/", "__")
        )
        self.model_source: str = local_path if os.path.isdir(local_path) else self.model_id

    # ── 내부 헬퍼 ──

    def _resolve_device(self) -> str:
        """요청된 디바이스와 실제 CUDA 유무를 비교해 최종 디바이스 결정."""
        if self.requested_device == "auto":
            return "cuda" if self.has_cuda else "cpu"
        if self.requested_device == "cuda" and not self.has_cuda:
            return "cpu"
        return self.requested_device

    @staticmethod
    def _require_int(key: str) -> int:
        """필수 정수 환경변수를 읽고, 없거나 잘못되면 명확한 에러를 던진다."""
        val = os.getenv(key)
        if val is None:
            raise RuntimeError(
                f"[CONFIG ERROR] 환경변수 '{key}' 가 설정되지 않았습니다.\n"
                f"  → jetson_slm_stack/.env 에 추가 후 컨테이너를 재시작하세요.\n"
                f"  → 예시: {key}=512"
            )
        try:
            return int(val)
        except ValueError:
            raise RuntimeError(
                f"[CONFIG ERROR] 환경변수 '{key}={val}' 가 유효한 정수가 아닙니다.\n"
                f"  → jetson_slm_stack/.env 를 수정 후 재시작하세요."
            )


# ═══════════════════════════════════════════════════════════════
# 3. MemoryManager — GPU/CPU 메모리 관리
# ═══════════════════════════════════════════════════════════════
class MemoryManager:
    """GPU/CPU 메모리 정리 및 OOM 에러 감지 유틸리티."""

    OOM_PATTERNS = (
        "out of memory",
        "cuda out of memory",
        "nvmapmemallocinternaltagged",
        "nvmapmemhandlealloc",
        "cannot allocate memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )

    def __init__(self, empty_cache_on_oom: bool, gc_interval: int = 0) -> None:
        self.empty_cache_on_oom = empty_cache_on_oom
        self.gc_interval = gc_interval
        self.cleanup_count = 0
        self.runtime_device = "cpu"

    def set_runtime_device(self, runtime_device: str) -> None:
        self.runtime_device = runtime_device

    def cleanup(self, force_gc: bool = False) -> None:
        """일반적인 메모리 정리 (gc + CUDA 캐시)."""
        self.cleanup_count += 1
        if force_gc or (self.gc_interval > 0 and self.cleanup_count % self.gc_interval == 0):
            gc.collect()
        if self.runtime_device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    def cleanup_after_oom(self) -> None:
        """OOM 발생 후 공격적 메모리 해제."""
        gc.collect()
        if self.runtime_device == "cuda" and torch.cuda.is_available() and self.empty_cache_on_oom:
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @classmethod
    def is_memory_error(cls, exc: BaseException) -> bool:
        """예외 메시지로부터 OOM 여부를 판별한다."""
        msg = str(exc).lower()
        return any(p in msg for p in cls.OOM_PATTERNS)


def _configure_cpu_runtime(cfg: AppConfig) -> Dict[str, int]:
    """CPU 추론용 스레드 수를 설정한다. C++ 확장이 있으면 그 경로를 우선 사용."""
    if _HAS_CPP:
        intra_threads, interop_threads = _cpp.configure_runtime(
            cfg.cpu_threads,
            cfg.cpu_interop_threads,
        )
    else:
        torch.set_num_threads(cfg.cpu_threads)
        try:
            torch.set_num_interop_threads(cfg.cpu_interop_threads)
        except RuntimeError:
            pass
        intra_threads = torch.get_num_threads()
        interop_threads = torch.get_num_interop_threads()
    return {
        "cpu_threads": int(intra_threads),
        "cpu_interop_threads": int(interop_threads),
    }
    def is_memory_error(cls, exc: BaseException) -> bool:
        """예외 메시지로부터 OOM 여부를 판별한다."""
        msg = str(exc).lower()
        return any(p in msg for p in cls.OOM_PATTERNS)


# ═══════════════════════════════════════════════════════════════
# 4. DtypeResolver — dtype 변환 유틸리티
# ═══════════════════════════════════════════════════════════════
class DtypeResolver:
    """문자열 dtype 이름 ↔ torch.dtype 변환."""

    _MAPPING = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }

    @classmethod
    def resolve(cls, name: str) -> torch.dtype:
        key = name.lower()
        if key not in cls._MAPPING:
            raise ValueError(f"지원하지 않는 dtype: {name}")
        return cls._MAPPING[key]

    @classmethod
    def pick_runtime(cls, device: str, requested: str) -> torch.dtype:
        """디바이스에 맞는 실행 시 dtype 결정. CPU에서 half는 float32로 승격."""
        dtype = cls.resolve(requested)
        if device != "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    @classmethod
    def get_autocast_context(cls, device: str, dtype: torch.dtype):
        """추론 시 사용할 autocast 컨텍스트를 반환한다."""
        if device == "cuda":
            return torch.autocast(device_type="cuda", dtype=dtype)
        if dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cpu", dtype=dtype)
        return nullcontext()


# ═══════════════════════════════════════════════════════════════
# 5. ModelLoader — 토크나이저와 모델 로드
# ═══════════════════════════════════════════════════════════════
class ModelLoader:
    """
    모델과 토크나이저를 로드하는 클래스.

    로드 전략 (우선순위):
      1) 4-bit 양자화 (CUDA 필수, bitsandbytes 필요)
      2) CPU 로드 → CUDA 이동 (Jetson nvmap 리크 방지)
      3) CPU float32 폴백
    """

    def __init__(self, cfg: AppConfig, mem: MemoryManager) -> None:
        self.cfg = cfg
        self.mem = mem

    # ── 토크나이저 ──

    def build_tokenizer(self) -> Any:
        tok = AutoTokenizer.from_pretrained(
            self.cfg.model_source, trust_remote_code=True, use_fast=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        return tok

    # ── 4-bit 양자화 설정 ──

    def _build_quant_config(self) -> Optional["BitsAndBytesConfig"]:
        if not (self.cfg.load_in_4bit and self.cfg.device == "cuda" and HAS_BNB_CONFIG):
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DtypeResolver.resolve(self.cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.cfg.bnb_4bit_use_double_quant,
        )

    # ── 모델 로드 ──

    def load_model(self) -> Tuple[Any, str, str, bool]:
        """
        모델을 로드하고 (model, active_device, load_mode, use_4bit) 튜플을 반환한다.
        """
        self.mem.cleanup()

        runtime_dtype = DtypeResolver.pick_runtime(self.cfg.device, self.cfg.dtype_name)
        quant_config = self._build_quant_config()
        use_4bit = quant_config is not None

        if use_4bit:
            model_obj, active_device, load_mode = self._load_4bit(quant_config)
            if model_obj is not None:
                return self._finalize(model_obj, active_device, load_mode, True)
            # 4-bit 실패 → CPU 폴백
            use_4bit = False
            runtime_dtype = torch.float32

        if self.cfg.device == "cuda" and self.cfg.has_cuda:
            offloaded = self._load_cuda_offload(runtime_dtype)
            if offloaded is not None:
                model_obj, active_device, load_mode = offloaded
            else:
                model_obj, active_device, load_mode = self._load_cuda(runtime_dtype)
        else:
            model_obj, active_device, load_mode = self._load_cpu()

        return self._finalize(model_obj, active_device, load_mode, use_4bit)

    def _load_4bit(self, quant_config) -> Tuple[Optional[Any], str, str]:
        """4-bit 양자화 로드 시도. 실패 시 (None, ...) 반환."""
        try:
            print("[INFO] 4-bit 양자화로 모델 로드 중...")
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_source,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=quant_config,
                device_map="auto",
            )
            print("[INFO] 4-bit 모델 로드 성공")
            return model_obj, "cuda", "4bit"
        except Exception as e:
            print(f"[WARN] 4-bit 로드 실패: {e}, CPU fp32 폴백")
            self.mem.cleanup_after_oom()
            return None, "cpu", "cpu-float32"

    def _load_cuda_offload(self, runtime_dtype: torch.dtype) -> Optional[Tuple[Any, str, str]]:
        """작은 GPU 메모리 예산으로 일부 레이어만 CUDA에 올리는 하이브리드 로드."""
        if not (self.cfg.gpu_offload_enabled and runtime_dtype == torch.float16):
            return None

        max_memory, budget_mb = self._build_cuda_offload_max_memory()
        if max_memory is not None:
            print(f"[INFO] CUDA 오프로딩 로드 시도 (gpu_budget={budget_mb}MiB)...")
            try:
                model_obj = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_source,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=runtime_dtype,
                    device_map="auto",
                    max_memory=max_memory,
                    offload_buffers=self.cfg.gpu_offload_buffers,
                )
                hf_map = getattr(model_obj, "hf_device_map", {})
                uses_cuda = any(
                    (isinstance(value, int))
                    or (isinstance(value, str) and value not in ("cpu", "disk"))
                    for value in hf_map.values()
                )
                if uses_cuda:
                    print(f"[INFO] CUDA 오프로딩 로드 성공 — budget={budget_mb}MiB")
                    return model_obj, "cuda", f"cuda-offload-{budget_mb}mb"

                print("[WARN] 오프로딩 로드가 GPU 레이어를 배치하지 못함")
                del model_obj
                self.mem.cleanup(force_gc=True)
            except Exception as e:
                print(f"[WARN] CUDA 오프로딩 로드 실패 ({e})")
                self.mem.cleanup_after_oom()

        if self.cfg.gpu_fixed_split_enabled:
            fixed_plan = self._probe_llama_cuda_split_plan()
            if fixed_plan is not None:
                budget_mb, gpu_layers = fixed_plan
                fixed_model = self._load_cuda_fixed_split(runtime_dtype, budget_mb, gpu_layers)
                if fixed_model is not None:
                    return fixed_model

        return self._load_cpu_bf16()

    def _load_cuda_fixed_split(
        self, runtime_dtype: torch.dtype, budget_mb: int, gpu_layers: int
    ) -> Optional[Tuple[Any, str, str]]:
        """Llama를 마지막 몇 개 레이어만 GPU에 올리는 고정 하이브리드 맵으로 로드한다."""
        device_map = self._build_llama_fixed_device_map(gpu_layers)
        if device_map is None:
            return None

        split_dtype = torch.bfloat16 if self._is_llama_model() else runtime_dtype

        print(
            "[INFO] 고정 CUDA 하이브리드 로드 시도 "
            f"(gpu_budget={budget_mb}MiB, gpu_layers={gpu_layers}, dtype={split_dtype})..."
        )
        try:
            model_obj = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_source,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=split_dtype,
                device_map=device_map,
            )
            hf_map = getattr(model_obj, "hf_device_map", {})
            uses_cuda = any(value == 0 or value == "cuda:0" for value in hf_map.values())
            if uses_cuda:
                print(
                    "[INFO] 고정 CUDA 하이브리드 로드 성공 "
                    f"— budget={budget_mb}MiB, gpu_layers={gpu_layers}, dtype={split_dtype}"
                )
                return model_obj, "cuda", f"cuda-fixed-{gpu_layers}l-{budget_mb}mb"

            print("[WARN] 고정 CUDA 하이브리드 로드가 GPU 레이어를 배치하지 못함")
            del model_obj
            self.mem.cleanup(force_gc=True)
            return None
        except Exception as e:
            print(f"[WARN] 고정 CUDA 하이브리드 로드 실패 ({e})")
            self.mem.cleanup_after_oom()
            return None

    def _load_cuda(self, runtime_dtype: torch.dtype) -> Tuple[Any, str, str]:
        """CPU 로드 후 CUDA 이동. 실패 시 bf16 CPU 폴백."""
        avail_mb = self._check_available_ram()
        weight_mb = self._estimate_weight_mb()

        if runtime_dtype == torch.float16 and weight_mb > 0:
            single_load_needed = weight_mb + self.cfg.cpu_single_load_margin_mb

            if avail_mb < single_load_needed:
                raise RuntimeError(
                    f"RAM 부족 ({avail_mb}MB < {single_load_needed}MB). "
                    f"현재 모델 단일 로드에도 메모리가 부족합니다."
                )

        print(f"[INFO] CPU로 모델 로드 (dtype={runtime_dtype})...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_source,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=runtime_dtype,
        )
        print("[INFO] CPU 로드 성공")
        load_mode = str(runtime_dtype).replace("torch.", "")

        # CUDA 이동 시도
        print("[INFO] 모델을 CUDA로 이동 중...")
        self.mem.cleanup()
        try:
            model_obj = model_obj.to("cuda")
            print("[INFO] CUDA 이동 성공")
            return model_obj, "cuda", load_mode
        except Exception as e:
            print(f"[WARN] CUDA 이동 실패 ({e})")
            self.mem.cleanup_after_oom()
            # fp16 모델 해제 → bf16으로 재로드 (변환 피크메모리 없음)
            del model_obj
            gc.collect()
            self.mem.cleanup()
            return self._load_cpu_bf16()

    def _load_cpu(self) -> Tuple[Any, str, str]:
        """CPU float32 전용 로드."""
        print("[INFO] CPU 모델 로드 (dtype=float32)...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_source,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        print("[INFO] CPU 로드 성공")
        return model_obj, "cpu", "cpu-float32"

    def _load_cpu_bf16(self) -> Tuple[Any, str, str]:
        """CPU bfloat16 로드. float16과 동일 메모리, float32 지수범위로 overflow 없음."""
        print("[INFO] CPU bfloat16 모델 로드...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_source,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        print("[INFO] CPU bfloat16 로드 성공")
        return model_obj, "cpu", "cpu-bfloat16"

    def _estimate_weight_mb(self) -> int:
        """로컬 모델 가중치 파일 크기를 MB 단위로 추정한다."""
        if not os.path.isdir(self.cfg.model_source):
            return 0

        candidates = []
        for name in os.listdir(self.cfg.model_source):
            if name.endswith((".safetensors", ".bin", ".pth")):
                candidates.append(os.path.join(self.cfg.model_source, name))

        if not candidates:
            return 0

        total_bytes = 0
        for path in candidates:
            try:
                total_bytes += os.path.getsize(path)
            except OSError:
                pass

        return total_bytes // (1024 * 1024)

    def _get_cuda_mem_info_mb(self) -> Tuple[int, int]:
        """CUDA free/total 메모리를 MB 단위로 반환한다."""
        free_bytes = 0
        total_bytes = 0
        try:
            if _HAS_CPP and hasattr(_cpp, "get_cuda_mem_info"):
                free_bytes, total_bytes = _cpp.get_cuda_mem_info()
            elif torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            pass
        return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))

    def _probe_cuda_budget_mb(self) -> Tuple[int, int, int]:
        """실제 cudaMalloc 성공값 기준의 안전한 GPU 예산을 찾는다."""
        free_mb, total_mb = self._get_cuda_mem_info_mb()
        safe_budget_mb = 0

        try:
            if _HAS_CPP and hasattr(_cpp, "probe_cuda_budget"):
                probed_free_mb, probed_total_mb, safe_budget_mb = _cpp.probe_cuda_budget(
                    self.cfg.gpu_target_memory_mb,
                    self.cfg.gpu_memory_reserve_mb,
                    self.cfg.gpu_probe_step_mb,
                    self.cfg.gpu_probe_min_mb,
                )
                free_mb = int(probed_free_mb)
                total_mb = int(probed_total_mb)
                safe_budget_mb = int(safe_budget_mb)
        except Exception:
            pass

        return free_mb, total_mb, safe_budget_mb

    def _probe_llama_cuda_split_plan(self) -> Optional[Tuple[int, int]]:
        """Llama용 고정 GPU split 계획을 네이티브 프로브 결과로 계산한다."""
        if not self._is_llama_model():
            return None

        total_layers = self._get_model_hidden_layers()
        if total_layers <= 0:
            return None

        try:
            if _HAS_CPP and hasattr(_cpp, "probe_llama_cuda_split"):
                free_mb, total_mb, safe_budget_mb, gpu_layers = _cpp.probe_llama_cuda_split(
                    self.cfg.gpu_target_memory_mb,
                    self.cfg.gpu_memory_reserve_mb,
                    self.cfg.gpu_probe_step_mb,
                    self.cfg.gpu_probe_min_mb,
                    total_layers,
                )
                budget_mb = int(safe_budget_mb)
                gpu_layers = int(gpu_layers)
                if budget_mb > 0 and gpu_layers > 0:
                    print(
                        "[INFO] 고정 llama split 프로브: "
                        f"free={int(free_mb)}MB total={int(total_mb)}MB "
                        f"safe={budget_mb}MB gpu_layers={gpu_layers}"
                    )
                    return budget_mb, gpu_layers
        except Exception:
            pass

        return None

    def _build_llama_fixed_device_map(self, gpu_layers: int) -> Optional[Dict[str, Any]]:
        """임베딩/헤드는 CPU에 두고 마지막 몇 개 llama 레이어만 GPU에 둔다."""
        if gpu_layers <= 0 or not self._is_llama_model():
            return None

        total_layers = self._get_model_hidden_layers()
        if total_layers <= 0:
            return None

        gpu_layers = min(gpu_layers, total_layers)
        first_gpu_layer = total_layers - gpu_layers
        device_map: Dict[str, Any] = {
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
            "model.norm": 0,
        }

        for layer_idx in range(total_layers):
            device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx >= first_gpu_layer else "cpu"

        return device_map

    def _is_llama_model(self) -> bool:
        """현재 모델이 llama 계열인지 확인한다."""
        model_ref = f"{self.cfg.model_id} {self.cfg.model_source}".lower()
        return "llama" in model_ref

    def _get_model_hidden_layers(self) -> int:
        """로컬 config.json에서 hidden layer 수를 읽는다."""
        if not os.path.isdir(self.cfg.model_source):
            return 0

        config_path = os.path.join(self.cfg.model_source, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_json = json.load(config_file)
            return int(config_json.get("num_hidden_layers", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    def _build_cuda_offload_max_memory(self) -> Tuple[Optional[Dict[Any, str]], int]:
        """작은 GPU 메모리 예산용 max_memory 맵을 구성한다."""
        free_mb, total_mb, safe_budget_mb = self._probe_cuda_budget_mb()
        budget_mb = safe_budget_mb if safe_budget_mb > 0 else min(
            self.cfg.gpu_target_memory_mb,
            max(0, free_mb - self.cfg.gpu_memory_reserve_mb),
        )

        if budget_mb < 512:
            print(
                "[WARN] CUDA 오프로딩 예산 부족: "
                f"free={free_mb}MB total={total_mb}MB reserve={self.cfg.gpu_memory_reserve_mb}MB safe={safe_budget_mb}MB"
            )
            return None, 0

        print(
            "[INFO] CUDA 예산 프로브: "
            f"free={free_mb}MB total={total_mb}MB safe={safe_budget_mb}MB target={self.cfg.gpu_target_memory_mb}MB"
        )

        cpu_budget_mb = max(self._estimate_weight_mb() + self.cfg.cpu_single_load_margin_mb, 4096)
        return {
            0: f"{budget_mb}MiB",
            "cpu": f"{cpu_budget_mb}MiB",
        }, budget_mb

    def _finalize(
        self, model_obj: Any, active_device: str, load_mode: str, use_4bit: bool
    ) -> Tuple[Any, str, str, bool]:
        """eval 모드, KV 캐시, torch.compile 적용 후 반환."""
        model_obj.eval()
        if hasattr(model_obj.config, "use_cache"):
            model_obj.config.use_cache = True

        if (
            self.cfg.enable_torch_compile
            and hasattr(torch, "compile")
            and active_device == "cuda"
        ):
            try:
                model_obj = torch.compile(model_obj, mode="reduce-overhead", fullgraph=False)
                print("[INFO] torch.compile 활성화")
            except Exception as e:
                print(f"[WARN] torch.compile 비활성화: {e}")

        self.mem.cleanup()
        print(f"[INFO] 모델 준비 완료 — device={active_device}, mode={load_mode}")
        return model_obj, active_device, load_mode, use_4bit

    @staticmethod
    def _check_available_ram() -> int:
        """Jetson 환경에서 가용 RAM을 MB 단위로 반환하고 임계값 미만이면 에러를 던진다."""
        avail_mb = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        avail_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass
        print(f"[INFO] 가용 RAM: {avail_mb} MB")

        if 0 < avail_mb < 2500:
            raise RuntimeError(
                f"RAM 부족 ({avail_mb}MB < 2500MB). "
                "Jetson 재부팅으로 nvmap/CUDA 상태를 초기화하세요."
            )
        return avail_mb

    @staticmethod
    def resolve_model_device(model_obj: Any) -> torch.device:
        """모델이 실제로 올라간 디바이스를 반환한다."""
        hf_map = getattr(model_obj, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            for value in hf_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value not in ("cpu", "disk"):
                    return torch.device(value)
        try:
            return next(model_obj.parameters()).device
        except Exception:
            return torch.device("cpu")

    @staticmethod
    def resolve_input_device(model_obj: Any) -> torch.device:
        """입력 토큰이 처음 소비되는 임베딩 모듈의 디바이스를 반환한다."""
        hf_map = getattr(model_obj, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            for module_name in (
                "model.embed_tokens",
                "model.decoder.embed_tokens",
                "transformer.wte",
            ):
                if module_name in hf_map:
                    value = hf_map[module_name]
                    if isinstance(value, int):
                        return torch.device(f"cuda:{value}")
                    if isinstance(value, str) and value not in ("disk", "meta"):
                        return torch.device(value)

        return ModelLoader.resolve_model_device(model_obj)


# ═══════════════════════════════════════════════════════════════
# 6-a. Float32LogitsProcessor
#      CPU float16 모델에서 logit overflow → inf → NaN 방지.
#      scores 를 float32 로 캐스팅해 torch.multinomial 크래시 예방.
# ═══════════════════════════════════════════════════════════════


class _Float32LogitsProcessor(LogitsProcessor):
    """CPU float16 모델에서 logit overflow → NaN 방지. scores 를 float32 로 캐스팅."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores.float()


# ═══════════════════════════════════════════════════════════════
# 6-b. InferenceEngine — 프롬프트 구성 · 토큰화 · 텍스트 생성
# ═══════════════════════════════════════════════════════════════
class InferenceEngine:
    """토크나이저와 모델을 사용해 텍스트 추론을 수행한다."""

    def __init__(
        self,
        cfg: AppConfig,
        mem: MemoryManager,
        tokenizer: Any,
        model: Any,
        active_device: str,
    ) -> None:
        self.cfg = cfg
        self.mem = mem
        self.tokenizer = tokenizer
        self.model = model
        self.active_device = active_device

        # CPU + float16 모델일 때만 logits processor 활성화
        # (bf16은 지수범위가 float32와 동일하므로 overflow 없음)
        self._needs_fp32_logits = (
            active_device != "cuda"
            and hasattr(model, "dtype")
            and model.dtype == torch.float16
        )
        if self._needs_fp32_logits:
            print("[INFO] CPU float16 모델 → Float32LogitsProcessor 활성화")

    # ── 프롬프트 구성 ──

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """채팅 메시지 리스트를 모델 입력 프롬프트 문자열로 변환한다."""
        rendered: List[Dict[str, str]] = []
        if not any(m["role"] == "system" for m in messages):
            rendered.append({"role": "system", "content": self.cfg.system_prompt})
        rendered.extend(messages)

        # chat_template 이 있으면 우선 사용
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    rendered, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass

        # 폴백: 단순 텍스트 포맷
        chunks = [f"[{m['role'].upper()}]\n{m['content']}" for m in rendered]
        chunks.append("[ASSISTANT]\n")
        return "\n\n".join(chunks)

    # ── 토큰화 ──

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """프롬프트를 토큰화하고 모델 디바이스로 옮긴다."""
        self.mem.cleanup()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_tokens,
            padding=False,
        )
        target_device = ModelLoader.resolve_input_device(self.model)
        return {k: v.to(target_device) for k, v in encoded.items()}

    # ── 샘플링 전략 ──

    def _choose_do_sample(self, temperature: float) -> bool:
        if (
            self.active_device != "cuda"
            and self.cfg.cpu_force_greedy
            and temperature <= self.cfg.cpu_force_greedy_max_temp
        ):
            return False
        if self.cfg.do_sample_default == "0":
            return False
        if self.cfg.do_sample_default == "1":
            return True
        return temperature > 0.0

    # ── 단일 생성 ──

    def generate_once(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        """한 번의 generate() 호출로 텍스트를 생성한다. OOM 시 축소 재시도."""
        self.mem.cleanup()

        encoded = self._prepare_inputs(prompt)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        runtime_dtype = DtypeResolver.pick_runtime(self.active_device, self.cfg.dtype_name)
        do_sample = self._choose_do_sample(temperature)

        gen_kwargs = self._build_gen_kwargs(
            encoded, max_new_tokens, do_sample, temperature, top_p, top_k, repetition_penalty,
        )

        started = time.time()
        output_ids = self._run_generate(gen_kwargs, runtime_dtype, max_new_tokens)
        latency = time.time() - started

        # 결과 디코딩
        new_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(new_ids.shape[0])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        self.mem.cleanup()
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_sec": round(latency, 3),
            "tokens_per_sec": round(completion_tokens / latency, 3) if latency > 0 else None,
        }

    def _build_gen_kwargs(
        self,
        encoded: Dict[str, torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        """model.generate() 에 전달할 인자 딕셔너리를 구성한다."""
        kwargs: Dict[str, Any] = {
            **encoded,
            "max_new_tokens": min(max_new_tokens, 1024),
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            kwargs["temperature"] = max(temperature, 1e-5)
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k if top_k > 0 else None

        # CPU float16 → float32 logits 캐스팅으로 NaN 방지
        if self._needs_fp32_logits:
            kwargs["logits_processor"] = [_Float32LogitsProcessor()]

        # None 값 제거 (generate() 경고 방지)
        return {k: v for k, v in kwargs.items() if v is not None}

    def _get_autocast(self, runtime_dtype: torch.dtype):
        """디바이스/dtype에 맞는 autocast 컨텍스트 반환."""
        return DtypeResolver.get_autocast_context(self.active_device, runtime_dtype)

    def _run_generate(
        self,
        kwargs: Dict[str, Any],
        runtime_dtype: torch.dtype,
        max_new_tokens: int,
    ) -> Any:
        """generate() 실행. OOM 시 축소된 토큰 수로 재시도한다."""
        try:
            with torch.inference_mode():
                with self._get_autocast(runtime_dtype):
                    output_ids = self.model.generate(**kwargs)
                if self.active_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
            return output_ids
        except RuntimeError as e:
            if not MemoryManager.is_memory_error(e):
                raise
            print("[WARN] 생성 중 OOM 발생, 축소 재시도...")
            self.mem.cleanup_after_oom()

            kwargs["max_new_tokens"] = max(16, min(max_new_tokens // 4, 64))
            with torch.inference_mode():
                with self._get_autocast(runtime_dtype):
                    output_ids = self.model.generate(**kwargs)
                if self.active_device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
            return output_ids

    # ── OOM 폴백 포함 생성 ──

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        """
        generate_once() 호출 후, OOM 발생 시 프롬프트 축소 + 토큰 축소로 재시도.
        """
        try:
            return self.generate_once(
                prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty,
            )
        except Exception as e:
            if not MemoryManager.is_memory_error(e):
                raise

            print(f"[WARN] 생성 메모리 부족: {e}")
            self.mem.cleanup_after_oom()

            retry_max = max(8, min(max_new_tokens // 4, 64))
            retry_prompt = self._try_truncate_prompt(prompt)

            print(f"[INFO] 축소 재시도: max_new_tokens={retry_max}, prompt_len≈{len(retry_prompt)}")
            try:
                result = self.generate_once(
                    retry_prompt, retry_max, temperature, top_p, top_k, repetition_penalty,
                )
                print("[INFO] 재시도 성공")
                return result
            except Exception as retry_e:
                print(f"[ERROR] 재시도도 실패: {retry_e}")
                self.mem.cleanup_after_oom()
                raise RuntimeError(
                    f"축소 재시도에서도 생성 실패: {retry_e}"
                )

    def _try_truncate_prompt(self, prompt: str) -> str:
        """프롬프트를 최대 토큰 수의 1/4 로 잘라 재시도 기회를 높인다."""
        try:
            trunc_len = max(64, self.cfg.max_input_tokens // 4)
            tokenized = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=trunc_len, padding=False,
            )
            truncated = self.tokenizer.decode(
                tokenized["input_ids"][0], skip_special_tokens=False,
            )
            print(f"[INFO] 프롬프트를 {trunc_len} 토큰으로 축소")
            return truncated
        except Exception as e:
            print(f"[WARN] 프롬프트 축소 실패: {e}")
            return prompt


# ═══════════════════════════════════════════════════════════════
# 7. API 스키마 (Pydantic) — 모듈 레벨 정의
#    cfg 를 먼저 생성해 default 값을 참조한다.
# ═══════════════════════════════════════════════════════════════

_cfg = AppConfig()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = Field(default=_cfg.max_new_tokens, ge=1, le=1024)
    temperature: float = Field(default=_cfg.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=_cfg.top_p, ge=0.0, le=1.0)
    top_k: int = Field(default=_cfg.top_k, ge=0, le=200)
    repetition_penalty: float = Field(default=_cfg.repetition_penalty, ge=1.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=_cfg.max_new_tokens, ge=1, le=1024)
    temperature: float = Field(default=_cfg.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=_cfg.top_p, ge=0.0, le=1.0)
    top_k: int = Field(default=_cfg.top_k, ge=0, le=200)
    repetition_penalty: float = Field(default=_cfg.repetition_penalty, ge=1.0, le=2.0)


# ═══════════════════════════════════════════════════════════════
# 8. FastAPI 앱 생성 + 라우트 등록
# ═══════════════════════════════════════════════════════════════

def create_app() -> Tuple[FastAPI, AppConfig]:
    """설정 → 메모리매니저 → 모델 로드 → FastAPI 앱 생성 → 라우트 등록."""

    # ── 초기화 (모듈 레벨 _cfg 재사용) ──
    cfg = _cfg
    runtime_info = _configure_cpu_runtime(cfg)
    mem = MemoryManager(cfg.empty_cache_on_oom, cfg.gc_interval)
    loader = ModelLoader(cfg, mem)
    tokenizer = loader.build_tokenizer()
    model, active_device, load_mode, use_4bit = loader.load_model()
    mem.set_runtime_device(active_device)
    engine = InferenceEngine(cfg, mem, tokenizer, model, active_device)

    # ── Lifespan 핸들러 (서버 시작/종료) ──
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        print("[INFO] ── 서버 시작 ──")
        print(f"  model_id        = {cfg.model_id}")
        print(f"  model_source    = {cfg.model_source}")
        print(f"  requested_device= {cfg.requested_device}")
        print(f"  active_device   = {active_device}")
        print(f"  load_mode       = {load_mode}")
        print(f"  4bit_requested  = {cfg.load_in_4bit}")
        print(f"  4bit_active     = {use_4bit}")
        print(f"  cpu_threads     = {runtime_info['cpu_threads']}")
        print(f"  cpu_interop     = {runtime_info['cpu_interop_threads']}")

        if cfg.enable_warmup:
            try:
                engine.generate_once(
                    prompt="Hello",
                    max_new_tokens=min(cfg.warmup_max_new_tokens, 4),
                    temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                )
                print("[INFO] 워밍업 성공")
            except Exception as e:
                print(f"[WARN] 워밍업 실패 (계속 진행): {e}")
                print("[DEBUG] traceback:", traceback.format_exc())
                mem.cleanup_after_oom()

        mem.cleanup(force_gc=True)
        print("[INFO] ── 서버 준비 완료 ──")
        yield
        mem.cleanup(force_gc=True)
        print("[INFO] ── 서버 종료 ──")

    app = FastAPI(title=cfg.app_title, lifespan=lifespan)

    # ── GET / — 서비스 상태 확인 ──
    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": cfg.app_title,
            "model": cfg.model_id,
            "status": "ok",
            "docs": "/docs",
        }

    # ── GET /healthz — 상세 헬스체크 ──
    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        cuda_mem = {}
        if torch.cuda.is_available():
            try:
                cuda_mem = {
                    "cuda_memory_allocated": int(torch.cuda.memory_allocated()),
                    "cuda_memory_reserved": int(torch.cuda.memory_reserved()),
                    "cuda_memory_total": int(torch.cuda.get_device_properties(0).total_memory),
                }
            except Exception:
                pass

        return {
            "ok": True,
            "model_id": cfg.model_id,
            "model_source": cfg.model_source,
            "requested_device": cfg.requested_device,
            "active_device": active_device,
            "model_device": str(ModelLoader.resolve_model_device(model)),
            "dtype": str(DtypeResolver.pick_runtime(active_device, cfg.dtype_name)),
            "cuda_available": torch.cuda.is_available(),
            "load_mode": load_mode,
            "load_in_4bit_requested": cfg.load_in_4bit,
            "load_in_4bit_active": use_4bit,
            "cpu_threads": runtime_info["cpu_threads"],
            "cpu_interop_threads": runtime_info["cpu_interop_threads"],
            "bnb_config_available": HAS_BNB_CONFIG,
            **cuda_mem,
            "max_input_tokens": cfg.max_input_tokens,
            "max_new_tokens_default": cfg.max_new_tokens,
        }

    # ── GET /v1/models — OpenAI-compatible 모델 목록 ──
    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "data": [{"id": cfg.model_id, "object": "model", "owned_by": "local"}]
        }

    # ── POST /generate — 단순 텍스트 생성 ──
    @app.post("/generate")
    def generate(req: GenerateRequest) -> Dict[str, Any]:
        result = engine.generate_text(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
        return {
            "model": cfg.model_id,
            "device": active_device,
            "load_mode": load_mode,
            **result,
        }

    # ── POST /v1/chat/completions — OpenAI-compatible 채팅 API ──
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> Dict[str, Any]:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        started = time.time()
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = engine.build_prompt(messages)

        result = engine.generate_text(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
        latency = time.time() - started

        return {
            "id": f"chatcmpl-local-{int(started * 1000)}",
            "object": "chat.completion",
            "created": int(started),
            "model": cfg.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
            "latency_sec": round(latency, 3),
            "tokens_per_sec": result["tokens_per_sec"],
            "device": active_device,
            "load_mode": load_mode,
        }

    return app, cfg


# ═══════════════════════════════════════════════════════════════
# 9. CLI 엔트리포인트
# ═══════════════════════════════════════════════════════════════
app, _cfg = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Inference Server")
    parser.add_argument("--model-id", type=str, default=None, help="MODEL_ID 환경변수 오버라이드")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="바인딩 호스트")
    args = parser.parse_args()

    if args.model_id:
        os.environ["MODEL_ID"] = args.model_id

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")