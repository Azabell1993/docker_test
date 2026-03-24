kmu-scrc@kmuscrc-desktop:~/jiwoo_research_docker$ echo 'How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?' | ./test_slm.sh llama

  Using prompt: "How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?"

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
  "cuda_memory_allocated": 566224384,
  "cuda_memory_reserved": 616562688,
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
  "text": "Here's what I have so far:\n\n1. **LLM Model**: You've created an LLM model using a popular library like Hugging Face Transformers or PyTorch. This model is trained on a large dataset of text.\n\n2. **Model Serving**: You've chosen a model serving platform like TensorFlow Serving, AWS SageMaker, or Azure Machine Learning to deploy your trained model. For this example, let's use TensorFlow Serving.\n\n3. **Apache Spark Integration**: You've integrated Apache Spark with your model serving platform to enable real-time processing of data. This integration allows you to process data in parallel across multiple nodes in your cluster.\n\n4. **NVIDIA DGX Cluster**: You've set up an NVIDIA DGX cluster, which provides a high-performance computing environment for deep learning tasks. The DGX cluster includes NVIDIA GPUs, which are ideal for training and deploying large-scale machine learning models.\n\n5. **Physical AI Applications**: You're working on physical AI applications in the maritime industry, such as predictive maintenance, anomaly detection, or optimization of ship operations.\n\nTo architect a pipeline to deploy and fine-tune your LLM on an NVIDIA DGX cluster integrated with Apache Spark for physical AI applications in the maritime industry, follow these steps:\n\n**Step 1: Prepare the DGX Cluster**\n\n* Set up the NVIDIA DGX cluster with the required number of GPUs.\n* Configure the cluster to use the GPU acceleration for your LLM model.\n* Ensure that the cluster has sufficient memory and storage to handle the model serving and data processing requirements.\n\n**Step 2: Integrate Apache Spark with the DGX Cluster**\n\n* Use the Apache Spark API to integrate the DGX cluster with your model serving platform.\n* Create a Spark job that runs on the DGX cluster to process data and serve the LLM model.\n* Use the Spark MLlib library to fine-tune the LLM model on the data processed by the Spark job.\n\n**Step 3: Deploy the LLM Model**\n\n* Use the TensorFlow Serving API to deploy the trained LLM model on the DGX cluster.\n* Configure the model serving platform to use the DGX cluster as the backend.\n* Use the TensorFlow Serving API to manage the model serving instance and retrieve the serving URL.\n\n**Step 4: Fine-Tune the LLM Model**\n\n* Use the Apache Spark API to fine-tune the LLM model on the data processed by the Spark job.\n* Use the Spark MLlib library to tune hyperparameters and update the model weights.\n* Use the TensorFlow",
  "prompt_tokens": 35,
  "completion_tokens": 512,
  "total_tokens": 547,
  "latency_sec": 439.407,
  "tokens_per_sec": 1.165
}


--- [TEST] Chat Completions API ---
  Prompt: "How can I architect a pipeline to deploy and fine-tune an LLM on an NVIDIA DGX cluster integrated with Apache Spark for Physical AI applications in the maritime industry?"
{
  "id": "chatcmpl-local-1774339291739",
  "object": "chat.completion",
  "created": 1774339291,
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Architecting a pipeline for deploying and fine-tuning a Large Language Model (LLM) on a NVIDIA DGX cluster integrated with Apache Spark for maritime industry applications involves several steps:\n\n**Step 1: Choose a suitable LLM model**\n\n* Select a pre-trained LLM that is well-suited for maritime industry applications, such as:\n\t+ BERT (Bidirectional Encoder Representations from Transformers)\n\t+ RoBERTa (Robustly Optimized BERT Pretraining Approach)\n\t+ XLNet (eXtreme Language Model)\n* Consider using a model that has been fine-tuned for specific domains like maritime, such as:\n\t+ Maritime-specific tasks like vessel tracking, navigation, or cargo optimization\n\n**Step 2: Prepare the data**\n\n* Collect relevant data for your maritime industry applications, including:\n\t+ Text data from various sources like ship logs, weather reports, and vessel performance data\n\t+ Image and video data for visual features like vessel identification, navigation, and cargo handling\n* Preprocess the data by:\n\t+ Tokenizing text data into individual words or subwords\n\t+ Normalizing numerical data\n\t+ Creating a custom dataset for training and testing\n\n**Step 3: Set up the NVIDIA DGX cluster**\n\n* Create a new NVIDIA DGX cluster with:\n\t+ 4-8 GPUs (e.g., V100, A100, or P100)\n\t+ 16-32 GB of memory per GPU\n\t+ 1-2 nodes with 64 GB of RAM each\n* Configure the cluster with:\n\t+ NVIDIA NVLink or NVLink II for high-speed inter-GPU communication\n\t+ NVIDIA's TensorRT 5.0 or later for optimized inference and training\n\t+ Docker containers for easy management and deployment\n\n**Step 4: Integrate Apache Spark**\n\n* Install Apache Spark 3.x on the cluster\n* Configure Spark to use the NVIDIA DGX cluster as a distributed computing environment\n* Use Spark's built-in support for GPU acceleration for tasks like data processing, machine learning, and deep learning\n\n**Step 5: Fine-tune the LLM**\n\n* Use a fine-tuning framework like TensorFlow, PyTorch, or Hugging Face's Transformers to fine-tune the pre-trained LLM on your dataset\n* Train the model on the prepared dataset using a suitable optimizer and learning rate schedule\n* Monitor the model's performance on a validation set during training\n* Use techniques like early stopping, gradient clipping, and regularization"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 84,
    "completion_tokens": 512,
    "total_tokens": 596
  },
  "latency_sec": 334.057,
  "tokens_per_sec": 1.533,
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
  Allocated:                     540.0 MB / 7619.9 MB  (7.1%)
  Reserved:                      588.0 MB / 7619.9 MB  (7.7%)

[ /generate API ]
  Prompt Tokens:                 35 / 1024  (3.4%)
  Completion Tokens:             512 / 512  (100.0%)
  Total Tokens:                  547 / 1536  (35.6%)
  Latency:                       439.407 sec
  Throughput:                    1.165 tok/s

[ /v1/chat/completions API ]
  Prompt Tokens:                 84 / 1024  (8.2%)
  Completion Tokens:             512 / 512  (100.0%)
  Total Tokens:                  596 / 1536  (38.8%)
  Latency:                       334.057 sec
  Throughput:                    1.533 tok/s

════════════════════════════════════════════════