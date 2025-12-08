# src/retrain.py
"""
Retraining pipeline.

- Scans all CSV files inside data/new/
- Loads all labeled examples
- Prints a summary (original behavior)
- If there is data → trains a simple fine-tuned model
- Saves model + metrics to models/fine_tuned/

This version keeps the original logic (list_new_data_files, load_labeled_data)
while adding the simplest possible real fine-tuning.
"""

from pathlib import Path
from typing import List
import json

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .data import list_new_data_files, load_labeled_data


BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_OUT_DIR = Path("models/fine_tuned")
METRICS_PATH = Path("models/fine_tuned/metrics.json")



def print_retraining_plan(files: List[Path], dfs: List[pd.DataFrame]) -> None:
    """Print original summary of labeled examples per file."""
    print("")
    print("📄 Summary of new labeled data:")
    total_examples = 0

    for path, df in zip(files, dfs):
        n = len(df)
        total_examples += n
        print(f"   • {path} → {n} examples")

    if total_examples == 0:
        print("\n New files found, but they contain NO valid labeled examples.")
    else:
        print(f"\nTotal new labeled examples: {total_examples}\n")

    print("Planned retraining steps:")
    print("  1. Merge new CSV files")
    print("  2. Train/validation split (80/20)")
    print("  3. Fine-tune pretrained model")
    print("  4. Evaluate")
    print("  5. Save fine-tuned model")
    print("")


#helpers
def _load_all_dfs(files: List[Path]) -> List[pd.DataFrame]:
    """Load each CSV into a DataFrame using load_labeled_data."""
    dfs = []
    for path in files:
        texts, labels = load_labeled_data(path)
        df = pd.DataFrame({"text": texts, "label": labels}).dropna()
        if not df.empty:
            df["label"] = df["label"].astype(int)
        dfs.append(df)
    return dfs


def _compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

#retraining pipeline
def main() -> None:
    files = list_new_data_files()

    if not files:
        print("No new labeled data found in data/new/. Skipping retraining.")
        return

    print(f"Detected {len(files)} new CSV file(s).")
    for f in files:
        print(f"   - {f}")

    # Load CSVs
    dfs = _load_all_dfs(files)

    # Print original-style summary
    print_retraining_plan(files, dfs)

    # Merge all non-empty DataFrames
    merged = pd.concat([df for df in dfs if not df.empty], ignore_index=True)

    if merged.empty:
        print("❌ No usable labeled data. Retraining aborted.")
        return

    # Split
    split_idx = int(len(merged) * 0.8)
    train_df = merged.iloc[:split_idx].reset_index(drop=True)
    val_df = merged.iloc[split_idx:].reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=3
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    print("Tokenizing...")
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    remove_cols_train = [
        c for c in train_ds.column_names if c not in ["input_ids", "attention_mask", "label"]
    ]
    remove_cols_val = [
        c for c in val_ds.column_names if c not in ["input_ids", "attention_mask", "label"]
    ]
    train_ds = train_ds.remove_columns(remove_cols_train).with_format("torch")
    val_ds = val_ds.remove_columns(remove_cols_val).with_format("torch")

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fine-tuning for 1 epoch...")
    args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR),
        num_train_epochs=1, #to keep it fast
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print("Metrics:", metrics)

    print("Saving fine-tuned model and metrics...")
    trainer.save_model(str(MODEL_OUT_DIR))

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved → {MODEL_OUT_DIR}")
    print(f"Metrics saved → {METRICS_PATH}")


if __name__ == "__main__":
    main()
