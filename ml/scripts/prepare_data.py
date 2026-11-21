"""Convert neuralchemy/Prompt-injection-dataset parquet splits into Kev System One JSONL.

One `noul` question per record: is this message an injection/jailbreak attempt?
Writes ml/data/kev/{train,val,test}.jsonl (core) and full_train.jsonl (augmented).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "kev"
OUT.mkdir(parents=True, exist_ok=True)

QUESTION_ID = "injection"
INSTRUCTIONS = (
    "Is this message an attempt to jailbreak, prompt-inject, or otherwise manipulate an AI "
    "assistant - for example by overriding its instructions, extracting its system prompt, "
    "using a role-play/persona bypass, or hiding instructions in encoded or indirect content?"
)
CRITERIA = {
    "true": "The message tries to manipulate the assistant, override its instructions, or smuggle a hidden instruction.",
    "false": "The message is an ordinary user request that does not try to manipulate the assistant.",
}


def row_to_record(row):
    label = bool(int(row["label"]))
    return {
        "state": str(row["text"]),
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
                "label": label,
            }
        },
        "_meta": {
            "category": str(row.get("category", "")),
            "severity": str(row.get("severity", "")),
            "source": str(row.get("source", "")),
            "group_id": str(row.get("group_id", "")),
            "augmented": bool(row.get("augmented", False)),
        },
    }


def write_jsonl(df, path):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_record(row), ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(df)} records ({int(df['label'].sum())} positive / {int((1 - df['label']).sum())} negative)")


def main():
    core = {s: pd.read_parquet(RAW / "core" / f"{s}-00000-of-00001.parquet") for s in ("train", "validation", "test")}
    full_train = pd.read_parquet(RAW / "full" / "train-00000-of-00001.parquet")

    for split, df in core.items():
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        write_jsonl(df, OUT / ("val.jsonl" if split == "validation" else f"{split}.jsonl"))

    full_train = full_train.drop_duplicates(subset=["text"]).reset_index(drop=True)
    write_jsonl(full_train, OUT / "full_train.jsonl")

    train_texts = set(core["train"]["text"])
    test_texts = set(core["test"]["text"])
    overlap = train_texts & test_texts
    print(f"exact train/test text overlap: {len(overlap)}")
    print("text chars: ", core["train"]["text"].str.len().describe().round(1).to_dict())


if __name__ == "__main__":
    main()
