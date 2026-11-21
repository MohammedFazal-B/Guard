"""Benchmark classifier latency for a Kev checkpoint (4-bit or bf16) on N records."""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT.parent / "kev"))

from kev.data import load_records, materialize  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "kev" / "test.jsonl"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--quantize", default="4bit")
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    dev = "cuda"
    tok = load_tokenizer(args.base, revision=args.revision or None)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    m = DecisionModel(args.base, tok, dev, revision=args.revision or None, head_dim=256,
                      quantize=args.quantize if args.quantize != "none" else None, dtype=dtype)
    m.lm = PeftModel.from_pretrained(m.lm, args.ckpt, torch_device=dev)
    meta = torch.load(Path(args.ckpt) / "head.pt", map_location="cpu", weights_only=False)
    m.head.load_state_dict(meta["head"])
    m.head.temperature = meta["temperature"]
    m.eval()

    reqs = load_records(Path(args.data))[: args.n]
    with torch.no_grad():
        for i, req in enumerate(reqs):
            t = time.time()
            p = m.probs(m.encode(tok, materialize(req)))[0]
            dt = time.time() - t
            print(f"{i}: {dt*1000:.0f} ms  p(pos)={float(p[1]):.3f}  len={len(req['state'])}")
    print(f"vram {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
