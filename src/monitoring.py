# src/monitoring.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]

SENTIMENT_LOG = REPO_ROOT / "data" / "monitoring" / "sentiment_log.csv"
MODEL_EVAL_LOG = REPO_ROOT / "data" / "monitoring" / "model_eval.csv"


# ---------- write: used by predict.py ----------
def record_prediction(label: str, score: float, text: str) -> None:
    """
    Appende una riga al CSV di monitoring.
    Questo è chiamato da predict_single / predict_batch.
    """
    SENTIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

    file_exists = SENTIMENT_LOG.exists()
    with SENTIMENT_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "label", "score", "text"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": (label or "").lower(),  # normalize
                "score": float(score),
                "text": text,
            }
        )


# ---------- read helpers ----------
def _read_csv_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-limit:] if rows else []


# ---------- sentiment: for /stats ----------
def load_sentiment_series(limit: int = 500):
    rows = _read_csv_tail(SENTIMENT_LOG, limit)
    for r in rows:
        try:
            r["score"] = float(r["score"])
        except (KeyError, TypeError, ValueError):
            # Se score non è convertibile, metti None e non rompere /stats
            r["score"] = None
    return rows


def load_sentiment_summary() -> Dict[str, Any]:
    rows = _read_csv_tail(SENTIMENT_LOG, 10_000_000)  # “tutto”
    if not rows:
        return {}

    counts: Dict[str, int] = {}
    for r in rows:
        lab = (r.get("label") or "").lower()
        if lab:
            counts[lab] = counts.get(lab, 0) + 1

    return {"total_predictions": int(len(rows)), "by_label": counts}


def load_sentiment_counts() -> Dict[str, int]:
    """
    Contatori semplici (Grafana-friendly):
    sentiment_counts.positive / neutral / negative / total
    """
    summary = load_sentiment_summary()
    by_label = summary.get("by_label") or {}

    positive = int(by_label.get("positive", 0))
    neutral = int(by_label.get("neutral", 0))
    negative = int(by_label.get("negative", 0))

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total": int(positive + neutral + negative),
    }


# ---------- model eval: for /stats ----------
def load_model_eval_latest() -> Optional[Dict[str, Any]]:
    rows = _read_csv_tail(MODEL_EVAL_LOG, 1)
    if not rows:
        return None
    last = rows[0]
    # cast
    if "accuracy" in last and last["accuracy"] not in (None, ""):
        last["accuracy"] = float(last["accuracy"])
    if "macro_f1" in last and last["macro_f1"] not in (None, ""):
        last["macro_f1"] = float(last["macro_f1"])
    if "n_samples" in last and last["n_samples"] not in (None, ""):
        last["n_samples"] = int(float(last["n_samples"]))
    return last


def load_model_eval_series(limit: int = 200) -> List[Dict[str, Any]]:
    rows = _read_csv_tail(MODEL_EVAL_LOG, limit)
    for r in rows:
        if "accuracy" in r and r["accuracy"] not in (None, ""):
            r["accuracy"] = float(r["accuracy"])
        if "macro_f1" in r and r["macro_f1"] not in (None, ""):
            r["macro_f1"] = float(r["macro_f1"])
        if "n_samples" in r and r["n_samples"] not in (None, ""):
            r["n_samples"] = int(float(r["n_samples"]))
    return rows


def build_stats_payload() -> Dict[str, Any]:
    sentiment_counts = load_sentiment_counts()

    return {
        "total_requests": sentiment_counts.get("total", 0),
        "label_counts": sentiment_counts,
        "label_distribution": sentiment_counts,

        "sentiment": {
            "summary": load_sentiment_summary(),
            "series": load_sentiment_series(),
        },
        "sentiment_counts": sentiment_counts,

        "model_eval": {
            "latest": load_model_eval_latest(),
            "series": load_model_eval_series(),
        },
    }
