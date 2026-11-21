"""Read the locked test split once at the threshold chosen on validation."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from calibrate_threshold import metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/data/kev/test.jsonl"
SCORES = ROOT / "ml/reports/test-scores.jsonl"
REPORT = ROOT / "ml/reports/kev4b-test-selected-threshold.json"
THRESHOLD = ROOT / "ml/reports/threshold.json"


def main():
    threshold = float(json.loads(THRESHOLD.read_text(encoding="utf-8"))["threshold"])
    records = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    existing = [json.loads(line) for line in SCORES.open(encoding="utf-8")] if SCORES.exists() else []
    if len(existing) > len(records):
        raise ValueError("scores file has more rows than test data")
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
                print(f"scored {index + 1}/{len(records)} test rows", flush=True)
    labels = [row["label"] for row in existing]
    scores = [row["probability"] for row in existing]
    report = {"samples": len(labels), "threshold_source": "validation", "selected": metrics(labels, scores, threshold),
              "default": metrics(labels, scores, 0.5)}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
