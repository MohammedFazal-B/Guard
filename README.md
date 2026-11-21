# LLM Guard

Local prompt injection guard for a chat powered by Ollama Gemma 4. A fine-tuned Kev classifier scores each user message; LangGraph routes blocked messages to a guard response and allowed messages to Ollama. FastAPI stores conversations in a local SQLite database, and Next.js provides the chat interface.

## Included model

`ml/runs/kev4b-injection` contains the trained LoRA adapter, pointer head, tokenizer, and training metadata. The adapter is stored with Git LFS. It needs the upstream `Qwen/Qwen3.5-4B-Base` backbone at revision `1001bb4d826a52d1f399e183466143f4da7b741b`. The multi-gigabyte backbone is downloaded separately.

The selected validation threshold is in `ml/reports/threshold.json`. On 942 held-out test messages, the guard achieved 96.28% accuracy, 96.83% F1, 96.92% injection recall, and 4.62% false positive rate. It missed 17 injections, so it is not a complete security boundary. See `ml/reports/kev4b-test-selected-threshold.json` for the full result.

## Run on Windows

Requirements: Python 3.12 or 3.13, a CUDA GPU with enough VRAM for 4-bit Qwen 4B, Node.js, pnpm, and Ollama with `gemma4:e4b` installed.

```powershell
git lfs install
git lfs pull
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\kev -e .\backend bitsandbytes huggingface_hub
.\.venv\Scripts\python.exe ml\scripts\download_base.py
```

Start the API from the repository root:

```powershell
$env:GUARD_BASE = "$PWD\ml\models\qwen3.5-4b-base"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the UI in a second terminal:

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:3000`. The first guarded request loads the model onto the GPU and may take a minute. `GUARD_CHECKPOINT`, `GUARD_THRESHOLD`, `OLLAMA_MODEL`, `OLLAMA_URL`, `GUARD_DB`, and `GUARD_API_URL` can be set to change the defaults.

## API

- `GET /health` reports checkpoint readiness.
- `POST /classify` scores a message without forwarding it to Ollama.
- `POST /chat` accepts `{ "message": "...", "conversation_id": null }`.
- `GET /conversations/{id}` returns saved messages.

The API returns 503 when the guard or Ollama is unavailable and does not forward a message to Ollama if the guard cannot classify it. The `kev` directory contains the Apache-2.0 Kev source used for the classifier; see `kev/LICENSE` and its upstream project at https://github.com/jaredpalmer/kev.
