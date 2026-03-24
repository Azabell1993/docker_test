kmu-scrc@kmuscrc-desktop:~/jiwoo_research_docker$ echo 'Did King Sejong create Hangul while using a MacBook?' | ./test_slm.sh llama

  Using prompt: "Did King Sejong create Hangul while using a MacBook?"

  [Config] max_new_tokens=512  temperature=0.0  top_p=1.0

--- Server already running on port 8000, skipping startup ---
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
  "cuda_memory_allocated": 534905344,
  "cuda_memory_reserved": 549453824,
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
  "text": "?\nI think there may be some misinformation here.\n\nKing Sejong, the fourth king of the Joseon Dynasty in Korea, is indeed famous for creating Hangul, the Korean alphabet. However, he did not use a MacBook.\n\nHangul was created in 1443 during King Sejong's reign, and it was designed to be a phonetic alphabet that represents the sounds of the Korean language. The creation of Hangul was a significant achievement, as it was the first true alphabet in Korea and has since become an integral part of Korean culture.\n\nAs for the technology used by King Sejong, we can only speculate about what he might have used. However, it's unlikely that he would have used a MacBook, which was introduced in the late 1990s and didn't become widely available until the early 2000s.\n\nIn fact, the first computers used by the Joseon Dynasty were likely to be simple machines like the \"seungje\" or \"seongje,\" which were used for administrative and military purposes. These early computers were likely to be based on mechanical or electromechanical systems, rather than modern computer hardware.\n\nSo, to summarize, King Sejong created Hangul without using a MacBook, and his technology was likely more basic than what we see today.",
  "prompt_tokens": 13,
  "completion_tokens": 260,
  "total_tokens": 273,
  "latency_sec": 157.34,
  "tokens_per_sec": 1.652
}


--- [TEST] Chat Completions API ---
  Prompt: "Did King Sejong create Hangul while using a MacBook?"
{
  "id": "chatcmpl-local-1774337782187",
  "object": "chat.completion",
  "created": 1774337782,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "No, King Sejong, the founder of the Korean alphabet (Hangul), did not use a MacBook. In fact, the first version of Hangul was created in 1443 during the Goryeo Dynasty in Korea.\n\nKing Sejong was a scholar and poet who played a crucial role in creating the Hangul alphabet, which was designed to be a phonetic and logographic system that could represent sounds and words in a single character. He worked on the project from 1400 to 1450, and his most famous work is the \"Sejongmun\" (Sejong's Dictionary), which was published in 1443.\n\nMacBooks, as we know them today, were not invented until more than 600 years later, in the 1980s by Steve Jobs and Steve Wozniak at Apple Inc.\n\nSo, it's not possible for King Sejong to have used a MacBook to create Hangul!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 189,
    "total_tokens": 251
  },
  "latency_sec": 114.964,
  "tokens_per_sec": 1.644,
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
  Allocated:                     510.1 MB / 7619.9 MB  (6.7%)
  Reserved:                      524.0 MB / 7619.9 MB  (6.9%)

[ /generate API ]
  Prompt Tokens:                 13 / 1024  (1.3%)
  Completion Tokens:             260 / 512  (50.8%)
  Total Tokens:                  273 / 1536  (17.8%)
  Latency:                       157.34 sec
  Throughput:                    1.652 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 62 / 1024  (6.1%)
  Completion Tokens:             189 / 512  (36.9%)
  Total Tokens:                  251 / 1536  (16.3%)
  Latency:                       114.964 sec
  Throughput:                    1.644 tok/s

════════════════════════════════════════════════