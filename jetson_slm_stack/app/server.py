from __future__ import annotations

import argparse
import gc
import os
import time
import traceback
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Jetson / embedded PyTorch workaround ──────────────────────────────────────
# The Jetson NGC PyTorch build does not ship torch._C._distributed_c10d.
# transformers.generate() calls is_fsdp_managed_module() which triggers an
# import of torch.distributed.fsdp that raises ModuleNotFoundError and causes
# all inference requests to fail with HTTP 500.
#
# Two-step patch required:
#   1) transformers.integrations.fsdp — module-level reference
#   2) transformers.generation.utils  — already imported by reference via
#      `from transformers.integrations.fsdp import is_fsdp_managed_module`
#      so the module patch alone is NOT sufficient.
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
# ─────────────────────────────────────────────────────────────────────────────

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB_CONFIG = True
except Exception:
    BitsAndBytesConfig = None
    HAS_BNB_CONFIG = False

APP_TITLE = os.getenv("APP_TITLE", "slm-inference-service")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

REQUESTED_DEVICE = os.getenv("DEVICE", "auto").lower()
HAS_CUDA = torch.cuda.is_available()

if REQUESTED_DEVICE == "auto":
    DEVICE = "cuda" if HAS_CUDA else "cpu"
elif REQUESTED_DEVICE == "cuda" and not HAS_CUDA:
    DEVICE = "cpu"
else:
    DEVICE = REQUESTED_DEVICE

DTYPE = os.getenv("DTYPE", "float16" if DEVICE == "cuda" else "float32").lower()

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TOP_K = int(os.getenv("TOP_K", "40"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

DO_SAMPLE_DEFAULT = os.getenv("DO_SAMPLE_DEFAULT", "auto").lower()  # auto|0|1
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "0") == "1"
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16").lower()
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4").lower()
BNB_4BIT_USE_DOUBLE_QUANT = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "0") == "1"

ENABLE_WARMUP = os.getenv("ENABLE_WARMUP", "1") == "1"
WARMUP_MAX_NEW_TOKENS = int(os.getenv("WARMUP_MAX_NEW_TOKENS", "8"))
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "0") == "1"
EMPTY_CACHE_ON_OOM = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a domain assistant for marine DX physical AI and network systems.",
)

LOCAL_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, MODEL_ID.replace("/", "__"))
MODEL_SOURCE = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else MODEL_ID


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def pick_runtime_dtype(device: str, requested: str) -> torch.dtype:
    dtype = resolve_dtype(requested)
    if device != "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def cleanup_after_oom() -> None:
    gc.collect()
    if torch.cuda.is_available() and EMPTY_CACHE_ON_OOM:
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_memory_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    patterns = (
        "out of memory",
        "cuda out of memory",
        "nvmapmemallocinternaltagged",
        "nvmapmemhandlealloc",
        "cannot allocate memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )
    return any(p in msg for p in patterns)


def resolve_model_device(model_obj: Any) -> torch.device:
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


def get_autocast_context(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_quant_config(device: str) -> Optional["BitsAndBytesConfig"]:
    if not LOAD_IN_4BIT:
        return None
    if device != "cuda":
        return None
    if not HAS_BNB_CONFIG:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=resolve_dtype(BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def build_tokenizer() -> Any:
    tok = AutoTokenizer.from_pretrained(
        MODEL_SOURCE,
        trust_remote_code=True,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model() -> Tuple[Any, str, str, bool]:
    # Pre-cleanup to maximize available memory before loading
    cleanup_memory()

    runtime_dtype = pick_runtime_dtype(DEVICE, DTYPE)
    quant_config = build_quant_config(DEVICE)
    use_4bit = quant_config is not None

    if use_4bit:
        # 4-bit quantization: use device_map="auto" (handled by bitsandbytes)
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "quantization_config": quant_config,
            "device_map": "auto",
        }
        load_mode = "4bit"
        active_device = "cuda"

        try:
            print("[INFO] Loading model in 4-bit quantization...")
            model_obj = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE, **kwargs)
            print("[INFO] 4-bit model loaded successfully")
        except Exception as e:
            print(f"[WARN] 4-bit load failed: {e}, falling back to CPU fp32")
            cleanup_after_oom()
            use_4bit = False
            runtime_dtype = torch.float32
            active_device = "cpu"
            load_mode = "cpu-float32"
            model_obj = AutoModelForCausalLM.from_pretrained(
                MODEL_SOURCE,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
            )
    else:
        # Step 1: Always load onto CPU first — avoids Tegra iGPU kernel-level OOM
        # (NvMapMemAllocInternalTagged error 12) that occurs when device_map="auto"
        # tries to place tensors directly into CUDA during from_pretrained.
        print(f"[INFO] Loading model to CPU first (dtype={runtime_dtype})...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=runtime_dtype,
        )
        print("[INFO] CPU load successful")
        load_mode = str(runtime_dtype).replace("torch.", "")
        active_device = "cpu"

        # Step 2: Try moving to CUDA only if requested and GPU is available
        if DEVICE == "cuda" and HAS_CUDA:
            print("[INFO] Attempting to move model to CUDA...")
            cleanup_memory()
            try:
                model_obj = model_obj.to("cuda")
                active_device = "cuda"
                print("[INFO] Model moved to CUDA successfully")
            except Exception as e:
                print(f"[WARN] CUDA move failed ({e}), staying on CPU")
                cleanup_after_oom()
                # Convert to float32 on CPU: float16 on CPU has limited op support
                # and can cause dtype mismatch errors during inference
                if runtime_dtype in (torch.float16, torch.bfloat16):
                    print("[INFO] Converting model to float32 for CPU inference compatibility")
                    model_obj = model_obj.to(torch.float32)
                    load_mode = "cpu-float32"
                else:
                    load_mode = f"cpu-{str(runtime_dtype).replace('torch.', '')}"

    model_obj.eval()
    if hasattr(model_obj.config, "use_cache"):
        model_obj.config.use_cache = True

    if ENABLE_TORCH_COMPILE and hasattr(torch, "compile") and active_device == "cuda":
        try:
            model_obj = torch.compile(model_obj, mode="reduce-overhead", fullgraph=False)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile disabled: {e}")

    cleanup_memory()
    print(f"[INFO] Model ready — device={active_device}, mode={load_mode}")
    return model_obj, active_device, load_mode, use_4bit


tokenizer = build_tokenizer()
model, ACTIVE_DEVICE, ACTIVE_LOAD_MODE, ACTIVE_4BIT = load_model()


@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    print("[INFO] startup begin")
    print(f"[INFO] model_id={MODEL_ID}")
    print(f"[INFO] model_source={MODEL_SOURCE}")
    print(f"[INFO] requested_device={REQUESTED_DEVICE}")
    print(f"[INFO] active_device={ACTIVE_DEVICE}")
    print(f"[INFO] load_mode={ACTIVE_LOAD_MODE}")
    print(f"[INFO] load_in_4bit_requested={LOAD_IN_4BIT}")
    print(f"[INFO] load_in_4bit_active={ACTIVE_4BIT}")

    if ENABLE_WARMUP:
        try:
            _ = generate_once(
                prompt="Hello",
                max_new_tokens=min(WARMUP_MAX_NEW_TOKENS, 4),
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
            print("[INFO] warmup success")
        except Exception as e:
            print(f"[WARN] warmup failed, but continuing: {e}")
            print("[DEBUG] traceback:", traceback.format_exc())
            cleanup_after_oom()

    cleanup_memory()
    print("[INFO] startup done")

    yield  # server is running

    # --- shutdown ---
    cleanup_memory()
    print("[INFO] shutdown complete")


app = FastAPI(title=APP_TITLE, lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=1024)
    temperature: float = Field(default=TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=TOP_P, ge=0.0, le=1.0)
    top_k: int = Field(default=TOP_K, ge=0, le=200)
    repetition_penalty: float = Field(default=REPETITION_PENALTY, ge=1.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=1024)
    temperature: float = Field(default=TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=TOP_P, ge=0.0, le=1.0)
    top_k: int = Field(default=TOP_K, ge=0, le=200)
    repetition_penalty: float = Field(default=REPETITION_PENALTY, ge=1.0, le=2.0)


def build_prompt(messages: List[Message]) -> str:
    rendered: List[Dict[str, str]] = []
    if not any(m.role == "system" for m in messages):
        rendered.append({"role": "system", "content": SYSTEM_PROMPT})
    rendered.extend({"role": m.role, "content": m.content} for m in messages)

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                rendered,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    chunks: List[str] = []
    for m in rendered:
        chunks.append(f"[{m['role'].upper()}]\n{m['content']}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)


def prepare_inputs(prompt: str) -> Dict[str, torch.Tensor]:
    cleanup_memory()  # Pre-cleanup before tokenization
    
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    )
    target_device = resolve_model_device(model)
    return {k: v.to(target_device) for k, v in encoded.items()}


def choose_do_sample(temperature: float) -> bool:
    if DO_SAMPLE_DEFAULT == "0":
        return False
    if DO_SAMPLE_DEFAULT == "1":
        return True
    return temperature > 0.0


def generate_once(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    cleanup_memory()  # Pre-cleanup before generation
    
    encoded = prepare_inputs(prompt)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    runtime_dtype = pick_runtime_dtype(ACTIVE_DEVICE, DTYPE)
    do_sample = choose_do_sample(temperature)

    kwargs: Dict[str, Any] = {
        **encoded,
        "max_new_tokens": min(max_new_tokens, 1024),
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = max(temperature, 1e-5)
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k if top_k > 0 else None

    # Remove None values that would cause generate() warnings
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    started = time.time()
    try:
        with torch.inference_mode():
            with get_autocast_context(ACTIVE_DEVICE, runtime_dtype):
                output_ids = model.generate(**kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except RuntimeError as e:
        # If OOM during generation, try with smaller max_new_tokens
        if is_memory_error(e):
            print(f"[WARN] OOM during generation, retrying with smaller context...")
            cleanup_after_oom()
            
            # Reduce max_new_tokens and retry
            reduced_tokens = max(16, min(max_new_tokens // 4, 64))
            kwargs["max_new_tokens"] = reduced_tokens
            
            with torch.inference_mode():
                with get_autocast_context(ACTIVE_DEVICE, runtime_dtype):
                    output_ids = model.generate(**kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        else:
            raise

    latency = time.time() - started
    new_ids = output_ids[0][prompt_tokens:]
    completion_tokens = int(new_ids.shape[0])
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    cleanup_memory()  # Post-cleanup after generation
    
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_sec": round(latency, 3),
        "tokens_per_sec": round(completion_tokens / latency, 3) if latency > 0 else None,
    }


def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    try:
        return generate_once(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    except Exception as e:
        if not is_memory_error(e):
            raise

        print(f"[WARN] Generation memory failure: {e}")
        cleanup_after_oom()

        # Aggressive reduction: use shorter prompt and smaller max_tokens
        retry_max_new_tokens = max(8, min(max_new_tokens // 4, 64))
        retry_prompt = prompt

        # Try to reduce prompt length
        try:
            truncated_tokens = max(64, MAX_INPUT_TOKENS // 4)
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=truncated_tokens,
                padding=False,
            )
            retry_prompt = tokenizer.decode(
                tokenized["input_ids"][0],
                skip_special_tokens=False,
            )
            print(f"[INFO] Prompt truncated to {truncated_tokens} tokens")
        except Exception as trunc_e:
            print(f"[WARN] Could not truncate prompt: {trunc_e}")

        print(
            f"[INFO] Retrying with reduced context: "
            f"max_new_tokens={retry_max_new_tokens}, prompt_len≈{len(retry_prompt)}"
        )

        try:
            result = generate_once(
                prompt=retry_prompt,
                max_new_tokens=retry_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            print("[INFO] Retry generation succeeded")
            return result
        except Exception as retry_e:
            print(f"[ERROR] Retry also failed: {retry_e}")
            cleanup_after_oom()
            raise RuntimeError(
                f"Generation failed even after retry with reduced context: {retry_e}"
            )


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": APP_TITLE, "model": MODEL_ID, "status": "ok", "docs": "/docs"}

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    cuda_mem_alloc = None
    cuda_mem_reserved = None
    cuda_mem_total = None
    if torch.cuda.is_available():
        try:
            cuda_mem_alloc = int(torch.cuda.memory_allocated())
            cuda_mem_reserved = int(torch.cuda.memory_reserved())
            cuda_mem_total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass

    return {
        "ok": True,
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "requested_device": REQUESTED_DEVICE,
        "active_device": ACTIVE_DEVICE,
        "model_device": str(resolve_model_device(model)),
        "dtype": str(pick_runtime_dtype(ACTIVE_DEVICE, DTYPE)),
        "cuda_available": torch.cuda.is_available(),
        "load_mode": ACTIVE_LOAD_MODE,
        "load_in_4bit_requested": LOAD_IN_4BIT,
        "load_in_4bit_active": ACTIVE_4BIT,
        "bnb_config_available": HAS_BNB_CONFIG,
        "cuda_memory_allocated": cuda_mem_alloc,
        "cuda_memory_reserved": cuda_mem_reserved,
        "cuda_memory_total": cuda_mem_total,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens_default": MAX_NEW_TOKENS,
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ]
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    result = generate_text(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )
    return {
        "model": MODEL_ID,
        "device": ACTIVE_DEVICE,
        "load_mode": ACTIVE_LOAD_MODE,
        **result,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    started = time.time()
    prompt = build_prompt(req.messages)
    result = generate_text(
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
        "model": MODEL_ID,
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
        "device": ACTIVE_DEVICE,
        "load_mode": ACTIVE_LOAD_MODE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Inference Server")
    parser.add_argument("--model-id", type=str, default=None, help="Override MODEL_ID env var")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    # Allow CLI override of MODEL_ID
    if args.model_id:
        os.environ["MODEL_ID"] = args.model_id

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")