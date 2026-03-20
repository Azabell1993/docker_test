import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN")
cache_dir = os.environ.get("MODEL_CACHE_DIR", "./models")
llama_model = os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
deepseek_model = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

os.makedirs(cache_dir, exist_ok=True)

for model_id in [llama_model, deepseek_model]:
    local_dir = os.path.join(cache_dir, model_id.replace("/", "__"))
    print(f"Downloading {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=hf_token,
        local_dir_use_symlinks=False,
    )