"""Evaluate a finished Kev injection checkpoint on the untouched test partition.

Run only after training finishes; this loads another copy of the base on the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kev"))
from kev.data import load_records, materialize  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402

REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "ml/runs/kev4b-injection"))
    parser.add_argument("--data", default=str(ROOT / "ml/data/kev/test.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "ml/reports/kev4b-test.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "head.pt").exists():
        parser.error(f"checkpoint is incomplete: {checkpoint}")
    tok = load_tokenizer("Qwen/Qwen3.5-4B-Base", revision=REVISION)
    model = DecisionModel("Qwen/Qwen3.5-4B-Base", tok, "cuda", revision=REVISION,
                          head_dim=256, quantize="4bit", dtype=torch.bfloat16)
    model.lm = PeftModel.from_pretrained(model.lm, str(checkpoint), torch_device="cuda")
    meta = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=False)
    model.head.load_state_dict(meta["head"])
    model.head.temperature = meta["temperature"]
    model.eval()

    labels, probabilities = [], []
    for index, request in enumerate(load_records(Path(args.data)), start=1):
        labels.append(int(request["questions"]["injection"]["label"]))
        with torch.no_grad():
            probabilities.append(float(model.probs(model.encode(tok, materialize(request)))[0][1]))
        if index % 50 == 0:
            print(f"scored {index}", flush=True)
    predictions = [int(p >= args.threshold) for p in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    report = {
        "checkpoint": str(checkpoint), "data": args.data, "samples": len(labels),
        "threshold": args.threshold, "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
