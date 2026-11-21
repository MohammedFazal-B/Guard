from huggingface_hub import snapshot_download

if __name__ == "__main__":
    p = snapshot_download(
        "Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        local_dir="ml/models/qwen3.5-4b-base",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
    )
    print(p)
