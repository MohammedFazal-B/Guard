"""Smoke test: load the base in NF4 + the released Kev-4B adapter and score one record."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kev"))

from kev.data import load_records, materialize  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE = "Qwen/Qwen3.5-4B-Base"
REV = "1001bb4d826a52d1f399e183466143f4da7b741b"
CKPT = Path(__file__).resolve().parents[1] / "models" / "kev-4b"


def main():
    dev = "cuda"
    t0 = time.time()
    tok = load_tokenizer(BASE, revision=REV)
    m = DecisionModel(BASE, tok, dev, revision=REV, head_dim=256, quantize="4bit")
    m.lm = PeftModel.from_pretrained(m.lm, str(CKPT), torch_device=dev)
    meta = torch.load(CKPT / "head.pt", map_location="cpu", weights_only=False)
    m.head.load_state_dict(meta["head"])
    m.head.temperature = meta["temperature"]
    m.eval()
    print(f"loaded in {time.time() - t0:.1f}s, vram {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    reqs = load_records(Path(__file__).resolve().parents[1] / "data" / "kev" / "test.jsonl")
    rec = materialize(reqs[0])
    t1 = time.time()
    with torch.no_grad():
        enc = m.encode(tok, rec)
        p = m.probs(enc)[0]
    print(f"probs {p.tolist()} in {time.time() - t1:.2f}s; label {rec['questions'][0]['label']}; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
