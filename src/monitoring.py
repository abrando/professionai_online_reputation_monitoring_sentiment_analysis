# src/monitoring.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import csv
from collections import deque

# Constants
REPO_ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_LOG = REPO_ROOT / "data" / "monitoring" / "sentiment_log.csv"
MODEL_EVAL_LOG = REPO_ROOT / "data" / "monitoring" / "model_eval.csv"

SERIES_WINDOW_ROWS = 500
TREND_WINDOW_SIZE = 50
TREND_POINTS = 500


# Record one prediction to the sentiment log
def record_prediction(label: str, score: float, text: str) -> None:
    """Append one prediction row to the monitoring CSV."""
    SENTIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not SENTIMENT_LOG.exists()
    with SENTIMENT_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_utc", "label", "score", "text"])
        if new:
            w.writeheader()
        w.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": (label or "").lower().strip(),
                "score": float(score),
                "text": text,
            }
        )


# Helper function to stream CSV rows
def _rows(path: Path):
    """Stream CSV rows as dicts; empty iterator if missing/unreadable."""
    if not path.exists():
        return iter(())
    try:
        f = path.open("r", encoding="utf-8", newline="")
    except Exception:
        return iter(())

    def gen():
        with f:
            for r in csv.DictReader(f):
                yield r
    return gen()


# Helper function to convert string to float
def _f(x: Any) -> Optional[float]:
    try:
        return None if x in (None, "") else float(x)
    except Exception:
        return None


# Helper function to convert string to int
def _i(x: Any) -> Optional[int]:
    try:
        return None if x in (None, "") else int(float(x))
    except Exception:
        return None


# Full scan of sentiment log to build stats
def _scan_sentiment() -> Dict[str, Any]:
    """
    One full pass:
    - full-history counts
    - non-cumulative moment trend (net counts in last N events)
    - tail raw series
    """
    g = {"positive": 0, "neutral": 0, "negative": 0}
    total = 0

    win = {"positive": 0, "neutral": 0, "negative": 0}
    labels = deque(maxlen=TREND_WINDOW_SIZE)
    trend = deque(maxlen=TREND_POINTS)
    series = deque(maxlen=SERIES_WINDOW_ROWS)

    for r in _rows(SENTIMENT_LOG):
        lab = (r.get("label") or "").lower().strip()
        ts = r.get("timestamp_utc") or r.get("timestamp") or ""

        # full-history counts
        if lab in g:
            g[lab] += 1
            total += 1

        # rolling window (moment)
        if len(labels) == labels.maxlen:
            old = labels[0]
            if old in win:
                win[old] -= 1
        labels.append(lab)
        if lab in win:
            win[lab] += 1

        trend.append(
            {
                "timestamp_utc": ts,
                "positive_count": int(win["positive"]),
                "neutral_count": int(win["neutral"]),
                "negative_count": int(win["negative"]),
                "window_size": int(len(labels)),
            }
        )

        rr = dict(r)
        rr["label"] = lab
        rr["timestamp_utc"] = rr.get("timestamp_utc") or rr.get("timestamp") or ""
        rr["score"] = _f(rr.get("score"))
        series.append(rr)

    return {"g": g, "total": total, "trend": list(trend), "series": list(series)}


# Return the latest model evaluation result
def _model_eval_latest() -> Optional[Dict[str, Any]]:
    last = None
    for r in _rows(MODEL_EVAL_LOG):
        last = r
    if not last:
        return None
    last = dict(last)
    last["accuracy"] = _f(last.get("accuracy"))
    last["macro_f1"] = _f(last.get("macro_f1"))
    last["n_samples"] = _i(last.get("n_samples"))
    return last


# Return a series of model evaluation results
def _model_eval_series(limit: int = 200) -> List[Dict[str, Any]]:
    tail = deque(maxlen=limit)
    for r in _rows(MODEL_EVAL_LOG):
        rr = dict(r)
        rr["accuracy"] = _f(rr.get("accuracy"))
        rr["macro_f1"] = _f(rr.get("macro_f1"))
        rr["n_samples"] = _i(rr.get("n_samples"))
        tail.append(rr)
    return list(tail)


# Build the full stats payload for monitoring
def build_stats_payload() -> Dict[str, Any]:
    s = _scan_sentiment()
    return {
        "sentiment_counts": {**s["g"], "total": int(s["total"]), "scope": "full_history"},
        "sentiment_trend_moment": s["trend"],
        "sentiment_trend_window_size": TREND_WINDOW_SIZE,
        "sentiment": {
            "summary": {"total_predictions": int(s["total"]), "by_label": s["g"], "scope": "full_history"},
            "series": s["series"],
        },
        "model_eval": {"latest": _model_eval_latest(), "series": _model_eval_series()},
    }
