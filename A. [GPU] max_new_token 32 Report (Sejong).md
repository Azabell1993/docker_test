Server is up!
--- [TEST] Health Check ---
{
  "ok": true,
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "model_source": "./models/meta-llama__Llama-3.2-1B-Instruct",
  "requested_device": "cuda",
  "active_device": "cuda",
  "model_device": "cuda:0",
  "dtype": "torch.float16",
  "cuda_available": true,
  "load_mode": "cuda-offload-640mb",
  "load_in_4bit_requested": false,
  "load_in_4bit_active": false,
  "cpu_threads": 6,
  "cpu_interop_threads": 1,
  "bnb_config_available": true,
  "cuda_memory_allocated": 526385664,
  "cuda_memory_reserved": 528482304,
  "cuda_memory_total": 7990083584,
  "max_input_tokens": 1024,
  "max_new_tokens_default": 512
}


--- [TEST] Models List ---
{
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "owned_by": "local"
    }
  ]
}


--- [TEST] Generation API ---
  Prompt: "Did King Sejong create Hangul while using a MacBook?"
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "cuda-offload-640mb",
  "text": "?\nI think there may be some misinformation here.\n\nKing Sejong, the fourth king of the Joseon Dynasty in Korea, is indeed famous for creating Hangul",
  "prompt_tokens": 13,
  "completion_tokens": 32,
  "total_tokens": 45,
  "latency_sec": 20.338,
  "tokens_per_sec": 1.573
}


--- [TEST] Chat Completions API ---
  Prompt: "Did King Sejong create Hangul while using a MacBook?"
{
  "id": "chatcmpl-local-1774336823294",
  "object": "chat.completion",
  "created": 1774336823,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "No, King Sejong, the founder of the Korean alphabet (Hangul), did not use a MacBook. In fact, the first version of Hangul was"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 32,
    "total_tokens": 94
  },
  "latency_sec": 19.593,
  "tokens_per_sec": 1.638,
  "device": "cuda",
  "load_mode": "cuda-offload-640mb"
}


════════════════════════════════════════════════
             PERFORMANCE REPORT                 
════════════════════════════════════════════════

[ Model ]
  Model ID:                      meta-llama/Llama-3.2-1B-Instruct
  Model Source:                  ./models/meta-llama__Llama-3.2-1B-Instruct
  Active Device:                 cuda
  CUDA Available:                true
  dtype:                         torch.float16

[ Token Limits ]
  Max Input Tokens:              1024 tokens
  Max New Tokens (default):      512 tokens
  Max Total (in+new):            1536 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     502.0 MB / 7619.9 MB  (6.6%)
  Reserved:                      504.0 MB / 7619.9 MB  (6.6%)

[ /generate API ]
  Prompt Tokens:                 13 / 1024  (1.3%)
  Completion Tokens:             32 / 512  (6.2%)
  Total Tokens:                  45 / 1536  (2.9%)
  Latency:                       20.338 sec
  Throughput:                    1.573 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 62 / 1024  (6.1%)
  Completion Tokens:             32 / 512  (6.2%)
  Total Tokens:                  94 / 1536  (6.1%)
  Latency:                       19.593 sec
  Throughput:                    1.638 tok/s

════════════════════════════════════════════════