"""Choose a guard threshold on validation data, never on the test partition.

Uses the running local classifier so only one copy of Kev occupies the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/data/kev/val.jsonl"
SCORES = ROOT / "ml/reports/validation-scores.jsonl"
REPORT = ROOT / "ml/reports/validation-threshold.json"
THRESHOLD = ROOT / "ml/reports/threshold.json"


def metrics(labels, scores, threshold):
    predicted = [int(p >= threshold) for p in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fpr": float(fp / (fp + tn)), "recall": float(tp / (tp + fn)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    records = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    existing = [json.loads(line) for line in SCORES.open(encoding="utf-8")] if SCORES.exists() else []
    if len(existing) > len(records):
        raise ValueError("scores file has more rows than validation data")
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=180) as client, SCORES.open("a", encoding="utf-8") as out:
        for index, record in enumerate(records[len(existing):], start=len(existing)):
            response = client.post("/classify", json={"message": record["state"]})
            response.raise_for_status()
            row = {"index": index, "label": int(record["questions"]["injection"]["label"]),
                   "probability": response.json()["injection_probability"]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            existing.append(row)
            if (index + 1) % 50 == 0:
                print(f"scored {index + 1}/{len(records)} validation rows", flush=True)
    labels = [row["label"] for row in existing]
    scores = [row["probability"] for row in existing]
    default = metrics(labels, scores, 0.5)
    candidates = sorted({0.5, *scores})
    eligible = [value for value in candidates if value >= 0.5 and
                sum(p < value for p, y in zip(scores, labels) if y == 1) <= default["confusion_matrix"]["fn"]]
    chosen = metrics(labels, scores, max(eligible))
    report = {"samples": len(labels),
              "selection_rule": "highest threshold that does not increase validation false negatives versus 0.5",
              "default": default, "chosen": chosen}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    THRESHOLD.write_text(json.dumps({"threshold": chosen["threshold"], "source": "validation",
                                     "rule": report["selection_rule"]}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
