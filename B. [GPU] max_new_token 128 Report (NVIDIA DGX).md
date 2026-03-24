kmu-scrc@kmuscrc-desktop:~/jiwoo_research_docker$ echo 'How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?' | ./test_slm.sh llama

  Using prompt: "How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?"

  [Config] max_new_tokens=128  temperature=0.0  top_p=1.0

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
  Prompt: "How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?"
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "device": "cuda",
  "load_mode": "cuda-offload-640mb",
  "text": "Here's what I have so far:\n\n1. **LLM Model**: You've created an LLM model using a popular library like Hugging Face Transformers or PyTorch. This model is trained on a large dataset of text.\n\n2. **Model Serving**: You've chosen a model serving platform like TensorFlow Serving, AWS SageMaker, or Azure Machine Learning to deploy your trained model. For this example, let's use TensorFlow Serving.\n\n3. **Apache Spark Integration**: You've integrated Apache Spark with your model serving platform to enable real-time processing of data. This integration allows you to process data in parallel across multiple nodes in your",
  "prompt_tokens": 35,
  "completion_tokens": 128,
  "total_tokens": 163,
  "latency_sec": 77.147,
  "tokens_per_sec": 1.659
}


--- [TEST] Chat Completions API ---
  Prompt: "How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?"
{
  "id": "chatcmpl-local-1774338263825",
  "object": "chat.completion",
  "created": 1774338263,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Architecting a pipeline for deploying and fine-tuning a Large Language Model (LLM) on a NVIDIA DGX cluster integrated with Apache Spark for maritime industry applications involves several steps:\n\n**Step 1: Choose a suitable LLM model**\n\n* Select a pre-trained LLM that is well-suited for maritime industry applications, such as:\n\t+ BERT (Bidirectional Encoder Representations from Transformers)\n\t+ RoBERTa (Robustly Optimized BERT Pretraining Approach)\n\t+ XLNet (eXtreme Language Model)\n* Consider using a model that has been fine-tuned for specific domains like maritime,"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 84,
    "completion_tokens": 128,
    "total_tokens": 212
  },
  "latency_sec": 77.712,
  "tokens_per_sec": 1.647,
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
  Prompt Tokens:                 35 / 1024  (3.4%)
  Completion Tokens:             128 / 512  (25.0%)
  Total Tokens:                  163 / 1536  (10.6%)
  Latency:                       77.147 sec
  Throughput:                    1.659 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 84 / 1024  (8.2%)
  Completion Tokens:             128 / 512  (25.0%)
  Total Tokens:                  212 / 1536  (13.8%)
  Latency:                       77.712 sec
  Throughput:                    1.647 tok/s

════════════════════════════════════════════════