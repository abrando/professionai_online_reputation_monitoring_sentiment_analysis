#!/usr/bin/env python3
"""
Baseline evaluation on public dataset (TweetEval sentiment) and saving
metrics in data/monitoring/model_eval.csv (for Grafana via /stats).
Used to monitor any performance drift of the model in production.

"""

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "data" / "monitoring" / "model_eval.csv"

# main function: evaluate and append CSV row
def main(max_samples: int = 2000, batch_size: int = 32) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("tweet_eval", "sentiment")["test"]
    ds = ds.select(range(min(max_samples, len(ds))))

    texts = ds["text"]
    y_true = [ID2LABEL[int(x)] for x in ds["label"]]

    clf = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)  # CPU
    y_pred = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        out = clf(batch, truncation=True)
        y_pred.extend([o["label"].lower() for o in out])

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": MODEL_NAME,
        "dataset": "tweet_eval/sentiment:test",
        "n_samples": len(texts),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    # append CSV (create if missing)
    df_row = pd.DataFrame([row])
    if OUT_CSV.exists():
        df_row.to_csv(OUT_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(OUT_CSV, index=False)

    print(f"[OK] appended -> {OUT_CSV}")
    print(f"Accuracy={acc:.4f}  MacroF1={macro_f1:.4f}  n={len(texts)}")


if __name__ == "__main__":
    main()
