kmu-scrc@kmuscrc-desktop:~/jiwoo_research_docker$ echo 'Create the "Hello World" first program, write the C++ language.' | ./test_slm.sh llama

  Using prompt: "Create the "Hello World" first program, write the C++ language."

  [Config] max_new_tokens=512  temperature=0.2  top_p=0.9

--- Server already running on port 8000, skipping startup ---
--- [TEST] Health Check ---
{
  "ok": true,
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "model_source": "./models/meta-llama__Llama-3.2-1B-Instruct",
  "requested_device": "cuda",
  "active_device": "cpu",
  "model_device": "cpu",
  "dtype": "torch.float32",
  "cuda_available": true,
  "load_mode": "cpu-float16",
  "load_in_4bit_requested": false,
  "load_in_4bit_active": false,
  "bnb_config_available": true,
  "cuda_memory_allocated": 0,
  "cuda_memory_reserved": 0,
  "cuda_memory_total": 7990079488,
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
  Prompt: "Create the "Hello World" first program, write the C++ language."
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cpu",
  "load_mode": "cpu-float16",
  "text": "Here is the code:\n\n```cpp\n#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << \"Hello, World!\" << endl;\n    return 0;\n}\n```\n\nThis is a simple C++ program that prints \"Hello, World!\" to the console. The `#include <iostream>` directive includes the iostream header file, which provides input/output functions such as `cout`. The `using namespace std;` directive allows us to use standard library functions without qualifying them with the `std::` prefix. The `main()` function is the entry point of the program, where execution begins.\n\nTo compile and run this program, save it in a file named `hello.cpp`, then use the following commands:\n\n```bash\ng++ hello.cpp -o hello\n./hello\n```\n\nThe `g++` compiler is used to compile the C++ code into an executable file. The `-o hello` flag specifies the output file name, and the `./hello` command runs the compiled program. When you run the program, it will print \"Hello, World!\" to the console.",
  "prompt_tokens": 16,
  "completion_tokens": 226,
  "total_tokens": 242,
  "latency_sec": 75.915,
  "tokens_per_sec": 2.977
}


--- [TEST] Chat Completions API ---
  Prompt: "Create the "Hello World" first program, write the C++ language."
{
  "id": "chatcmpl-local-1773995089805",
  "object": "chat.completion",
  "created": 1773995089,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "**Hello World in C++**\n======================\n\nBelow is a simple \"Hello World\" program written in C++.\n\n```cpp\n// hello_world.cpp\n\n#include <iostream>\n\nint main() {\n    // Print \"Hello, World!\" to the console\n    std::cout << \"Hello, World!\" << std::endl;\n\n    return 0;\n}\n```\n\nTo compile and run this program:\n\n1. Save the above code in a file named `hello_world.cpp`.\n2. Open a terminal or command prompt.\n3. Navigate to the directory where you saved the file.\n4. Compile the program using the following command:\n   ```bash\ng++ -o hello_world hello_world.cpp\n```\n5. Run the program using the following command:\n   ```bash\n./hello_world\n```\n\nThis will output:\n```\nHello, World!\n```\n\n**Explanation**\n\n* The `#include <iostream>` directive includes the iostream library, which provides input/output functions such as `std::cout`.\n* The `main` function is the entry point of the program.\n* The `std::cout` object is used to print \"Hello, World!\" to the console.\n* The `<<` operator is used to insert the string \"Hello, World!\" into the output stream.\n* The `std::endl` object is used to insert a newline character at the end of the output.\n* The `return 0` statement indicates successful execution of the program."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 65,
    "completion_tokens": 301,
    "total_tokens": 366
  },
  "latency_sec": 106.406,
  "tokens_per_sec": 2.855,
  "device": "cpu",
  "load_mode": "cpu-float16"
}


════════════════════════════════════════════════
             PERFORMANCE REPORT
════════════════════════════════════════════════

[ Model ]
  Model ID:                      meta-llama/Llama-3.2-1B-Instruct
  Model Source:                  ./models/meta-llama__Llama-3.2-1B-Instruct
  Active Device:                 cpu
  CUDA Available:                true
  dtype:                         torch.float32

[ Token Limits ]
  Max Input Tokens:              1024 tokens
  Max New Tokens (default):      512 tokens
  Max Total (in+new):            1536 tokens

[ CUDA Memory  (used / total) ]
  Allocated:                     0.0 MB / 7619.9 MB  (0.0%)
  Reserved:                      0.0 MB / 7619.9 MB  (0.0%)

[ /generate API ]
  Prompt Tokens:                 16 / 1024  (1.6%)
  Completion Tokens:             226 / 512  (44.1%)
  Total Tokens:                  242 / 1536  (15.8%)
  Latency:                       75.915 sec
  Throughput:                    2.977 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 65 / 1024  (6.3%)
  Completion Tokens:             301 / 512  (58.8%)
  Total Tokens:                  366 / 1536  (23.8%)
  Latency:                       106.406 sec
  Throughput:                    2.855 tok/s

════════════════════════════════════════════════