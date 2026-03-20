kmu-scrc@kmuscrc-desktop:~/jiwoo_research_docker$ echo 'Did King Sejong create Hangul while using a MacBook?' | ./test_slm.sh llama

  Using prompt: "Did King Sejong create Hangul while using a MacBook?"

  [Config] max_new_tokens=512  temperature=0.2  top_p=0.9

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
  "load_mode": "float16",
  "load_in_4bit_requested": false,
  "load_in_4bit_active": false,
  "bnb_config_available": true,
  "cuda_memory_allocated": 2489717248,
  "cuda_memory_reserved": 2497708032,
  "cuda_memory_total": 7990075392,
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
  "load_mode": "float16",
  "text": "?\nI think there may be some misconceptions here.\n\nKing Sejong, the third king of the Joseon Dynasty in Korea, is indeed famous for creating Hangul, the Korean alphabet. However, he did not use a MacBook.\n\nHangul was created in 1443 during King Sejong's reign, and it was designed to be a phonetic alphabet that represents each character with a unique sound. The creation of Hangul was a significant achievement in Korean history, as it helped to standardize the Korean language and make it more accessible to the general population.\n\nAs for the technology used by King Sejong, we don't know exactly what he used, but it's likely that he relied on traditional writing materials such as ink, paper, and quills. There are no records or accounts from King Sejong himself that mention using a MacBook or any other modern computer.\n\nIn fact, the first computers were not even invented until the mid-20th century, more than 300 years after King Sejong's time. So, it's highly unlikely that he would have used a MacBook or any other modern device.\n\nSo, to summarize: King Sejong created Hangul in 1443 using traditional writing materials, and he didn't use a MacBook or any other modern computer.",
  "prompt_tokens": 13,
  "completion_tokens": 258,
  "total_tokens": 271,
  "latency_sec": 15.542,
  "tokens_per_sec": 16.6
}


--- [TEST] Chat Completions API ---
  Prompt: "Did King Sejong create Hangul while using a MacBook?"
{
  "id": "chatcmpl-local-1773996709213",
  "object": "chat.completion",
  "created": 1773996709,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "No, King Sejong the Great did not use a MacBook. He was a Korean king who ruled from 1418 to 1450 and is best known for creating the Korean alphabet, Hangul.\n\nHangul was actually created in 1443 during King Sejong's reign, but it was based on earlier writing systems such as the Chinese characters and the Japanese Kanji. The first version of Hangul was called \"Sejong's Script\" or \"Sejong's alphabet,\" and it was designed to be a phonetic alphabet that could be read by anyone, regardless of their native language or education level.\n\nKing Sejong's script was written in a unique combination of consonant-vowel pairs, which were chosen to be easy to pronounce and recognize. It was also designed to be a more efficient and practical alternative to existing writing systems, which often had many exceptions and complexities.\n\nThere is no evidence to suggest that King Sejong used a MacBook, as they were not widely available in Korea at the time. In fact, the first personal computers were not developed until the 1960s, more than 300 years after King Sejong's death."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 233,
    "total_tokens": 295
  },
  "latency_sec": 15.027,
  "tokens_per_sec": 16.553,
  "device": "cuda",
  "load_mode": "float16"
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
  Allocated:                     2374.4 MB / 7619.9 MB  (31.2%)
  Reserved:                      2382.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 13 / 1024  (1.3%)
  Completion Tokens:             258 / 512  (50.4%)
  Total Tokens:                  271 / 1536  (17.6%)
  Latency:                       15.542 sec
  Throughput:                    16.6 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 62 / 1024  (6.1%)
  Completion Tokens:             233 / 512  (45.5%)
  Total Tokens:                  295 / 1536  (19.2%)
  Latency:                       15.027 sec
  Throughput:                    16.553 tok/s

════════════════════════════════════════════════