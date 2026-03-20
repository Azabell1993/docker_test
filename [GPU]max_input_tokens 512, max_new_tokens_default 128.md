
  Using prompt: "What is the capital of France?"

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
  "cuda_memory_allocated": 2488668672,
  "cuda_memory_reserved": 2499805184,
  "cuda_memory_total": 7990079488,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
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
  Prompt: "What is the capital of France?"
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "Paris.\nThe capital of France is indeed Paris, a city known for its rich history, art museums, and iconic landmarks like the Eiffel Tower. It's also home to many famous artists, writers, and intellectuals throughout history.\n\nHere are some fun facts about Paris:\n\n1. **Paris is the most visited city",
  "prompt_tokens": 8,
  "completion_tokens": 64,
  "total_tokens": 72,
  "latency_sec": 3.93,
  "tokens_per_sec": 16.284
}


--- [TEST] Chat Completions API ---
  Prompt: "What is the capital of France?"
{
  "id": "chatcmpl-local-1773989546067",
  "object": "chat.completion",
  "created": 1773989546,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 57,
    "completion_tokens": 8,
    "total_tokens": 65
  },
  "latency_sec": 1.517,
  "tokens_per_sec": 13.82,
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
  Max Input Tokens:              512 tokens
  Max New Tokens (default):      128 tokens
  Max Total (in+new):            640 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     2373.4 MB / 7619.9 MB  (31.1%)
  Reserved:                      2384.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 8 / 512  (1.6%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  72 / 640  (11.2%)
  Latency:                       3.93 sec
  Throughput:                    16.284 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 57 / 512  (11.1%)
  Completion Tokens:             8 / 128  (6.2%)
  Total Tokens:                  65 / 640  (10.2%)
  Latency:                       1.517 sec
  Throughput:                    13.82 tok/s

════════════════════════════════════════════════

  Using prompt: "Create the 'Hello World' first program, write the C++ language."

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
  "cuda_memory_allocated": 2488668672,
  "cuda_memory_reserved": 2497708032,
  "cuda_memory_total": 7990079488,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
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
  Prompt: "Create the 'Hello World' first program, write the C++ language."
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "Here is a simple C++ program that prints \"Hello, World!\" to the console.\n\n```cpp\n#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}\n```\n\nThis code includes the necessary `#include` directive for the",
  "prompt_tokens": 16,
  "completion_tokens": 64,
  "total_tokens": 80,
  "latency_sec": 3.932,
  "tokens_per_sec": 16.278
}


--- [TEST] Chat Completions API ---
  Prompt: "Create the 'Hello World' first program, write the C++ language."
{
  "id": "chatcmpl-local-1773989553994",
  "object": "chat.completion",
  "created": 1773989553,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is the \"Hello World\" program in C++:\n\n```cpp\n// Hello World in C++\n#include <iostream>\n\nint main() {\n    // Print \"Hello, World!\" to the console\n    std::cout << \"Hello, World!\" << std::endl;\n\n    return 0;\n}\n```"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 65,
    "completion_tokens": 64,
    "total_tokens": 129
  },
  "latency_sec": 4.889,
  "tokens_per_sec": 16.228,
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
  Max Input Tokens:              512 tokens
  Max New Tokens (default):      128 tokens
  Max Total (in+new):            640 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     2373.4 MB / 7619.9 MB  (31.1%)
  Reserved:                      2382.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 16 / 512  (3.1%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  80 / 640  (12.5%)
  Latency:                       3.932 sec
  Throughput:                    16.278 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 65 / 512  (12.7%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  129 / 640  (20.2%)
  Latency:                       4.889 sec
  Throughput:                    16.228 tok/s

════════════════════════════════════════════════

  Using prompt: "Explain the difference between TCP and UDP in simple terms."

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
  "cuda_memory_allocated": 2488668672,
  "cuda_memory_reserved": 2499805184,
  "cuda_memory_total": 7990079488,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
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
  Prompt: "Explain the difference between TCP and UDP in simple terms."
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "Think of it like sending a letter to a friend.\n\nImagine you're writing a letter to your friend, and you want to make sure that the letter arrives safely and on time. You have two options: send it via a postal service (like USPS) or use a courier service (like UPS).\n\n**TCP (Transmission",
  "prompt_tokens": 13,
  "completion_tokens": 64,
  "total_tokens": 77,
  "latency_sec": 3.93,
  "tokens_per_sec": 16.286
}


--- [TEST] Chat Completions API ---
  Prompt: "Explain the difference between TCP and UDP in simple terms."
{
  "id": "chatcmpl-local-1773989565325",
  "object": "chat.completion",
  "created": 1773989565,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "In the context of computer networking, TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are two fundamental protocols that enable data communication over the internet. Here's a simplified explanation of the main differences between them:\n\n**TCP (Transmission Control Protocol)**\n\n* **Reliable**: TCP ensures that data is delivered in"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 62,
    "completion_tokens": 64,
    "total_tokens": 126
  },
  "latency_sec": 4.901,
  "tokens_per_sec": 16.23,
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
  Max Input Tokens:              512 tokens
  Max New Tokens (default):      128 tokens
  Max Total (in+new):            640 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     2373.4 MB / 7619.9 MB  (31.1%)
  Reserved:                      2384.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 13 / 512  (2.5%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  77 / 640  (12.0%)
  Latency:                       3.93 sec
  Throughput:                    16.286 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 62 / 512  (12.1%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  126 / 640  (19.7%)
  Latency:                       4.901 sec
  Throughput:                    16.23 tok/s

════════════════════════════════════════════════

  Using prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms."

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
  "cuda_memory_allocated": 2488668672,
  "cuda_memory_reserved": 2497708032,
  "cuda_memory_total": 7990079488,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
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
  Prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms."
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "The Fibonacci sequence is a series of numbers in which each number is the sum of the two preceding ones, usually starting with 0 and 1.\n\nHere's an example of how you can use this function:\n\n```python\ndef fibonacci(n):\n    fib_sequence = [0, 1]\n    while len(fib",
  "prompt_tokens": 15,
  "completion_tokens": 64,
  "total_tokens": 79,
  "latency_sec": 3.934,
  "tokens_per_sec": 16.268
}


--- [TEST] Chat Completions API ---
  Prompt: "Write a Python function that calculates the Fibonacci sequence up to n terms."
{
  "id": "chatcmpl-local-1773989576645",
  "object": "chat.completion",
  "created": 1773989576,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "**Fibonacci Sequence Function**\n================================\n\nThe Fibonacci sequence is a series of numbers in which each number is the sum of the two preceding ones, usually starting with 0 and 1.\n\nHere's a simple Python function that calculates the Fibonacci sequence up to `n` terms:\n\n```python\ndef fibonacci(n):"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 64,
    "completion_tokens": 64,
    "total_tokens": 128
  },
  "latency_sec": 4.846,
  "tokens_per_sec": 16.377,
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
  Max Input Tokens:              512 tokens
  Max New Tokens (default):      128 tokens
  Max Total (in+new):            640 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     2373.4 MB / 7619.9 MB  (31.1%)
  Reserved:                      2382.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 15 / 512  (2.9%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  79 / 640  (12.3%)
  Latency:                       3.934 sec
  Throughput:                    16.268 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 64 / 512  (12.5%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  128 / 640  (20.0%)
  Latency:                       4.846 sec
  Throughput:                    16.377 tok/s

════════════════════════════════════════════════

  Using prompt: "인공지능과 머신러닝의 차이점을 간단히 설명해줘."

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
  "cuda_memory_allocated": 2488668672,
  "cuda_memory_reserved": 2499805184,
  "cuda_memory_total": 7990079488,
  "max_input_tokens": 512,
  "max_new_tokens_default": 128
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
  Prompt: "인공지능과 머신러닝의 차이점을 간단히 설명해줘."
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "float16",
  "text": "인공지능은 인간의 지식과 경험을 바탕으로 machine learning을 통해 machine가 다른 machine를 예상할 수 있는 능력을 가지고 있다. 머신러닝은 machine learning의 한 종류로, machine learning을 통해 machine가 다른 machine를 예상할 수 있는 능력을 가지고 있다",
  "prompt_tokens": 22,
  "completion_tokens": 64,
  "total_tokens": 86,
  "latency_sec": 3.919,
  "tokens_per_sec": 16.33
}


--- [TEST] Chat Completions API ---
  Prompt: "인공지능과 머신러닝의 차이점을 간단히 설명해줘."
{
  "id": "chatcmpl-local-1773989587928",
  "object": "chat.completion",
  "created": 1773989587,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "인공지능 (AI)와 머신 러닝 (ML)은 두 가지 다른 기술로, 이 두 가지는 다르지만, 모두 인공智能을 구현하는 데 사용되는 기술입니다.\n\n**인공지능 (AI)**\n\n인공지능은 artificial intelligence (AI)라는 용어를 사용"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 71,
    "completion_tokens": 64,
    "total_tokens": 135
  },
  "latency_sec": 4.85,
  "tokens_per_sec": 16.336,
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
  Max Input Tokens:              512 tokens
  Max New Tokens (default):      128 tokens
  Max Total (in+new):            640 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     2373.4 MB / 7619.9 MB  (31.1%)
  Reserved:                      2384.0 MB / 7619.9 MB  (31.3%)

[ /generate API ]
  Prompt Tokens:                 22 / 512  (4.3%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  86 / 640  (13.4%)
  Latency:                       3.919 sec
  Throughput:                    16.33 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 71 / 512  (13.9%)
  Completion Tokens:             64 / 128  (50.0%)
  Total Tokens:                  135 / 640  (21.1%)
  Latency:                       4.85 sec
  Throughput:                    16.336 tok/s

════════════════════════════════════════════════
