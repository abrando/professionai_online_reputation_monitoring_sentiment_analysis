# src/monitoring.py
"""
Monitoring utilities.

This module implements a simple time-series style monitoring layer:
- each prediction is logged with a timestamp in a CSV file;
- an in-memory state is kept for quick access by the /stats endpoint;
- the /stats JSON is designed to be consumed by Grafana Infinity as
  a time series (`time_series` field).
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal
import csv


SentimentLabel = Literal["negative", "neutral", "positive"]


LOG_DIR = Path("data/monitoring")
LOG_FILE = LOG_DIR / "sentiment_log.csv"


@dataclass
class SentimentEvent:
    """Represents a single prediction event in time."""
    timestamp: datetime
    text_length: int
    label: SentimentLabel
    score: float


@dataclass
class MonitoringState:
    """Minimal in-memory monitoring state (process-local)."""
    total_requests: int = 0
    label_counts: Dict[SentimentLabel, int] = field(
        default_factory=lambda: {"negative": 0, "neutral": 0, "positive": 0}
    )
    recent_events: List[SentimentEvent] = field(default_factory=list)
    max_recent_events: int = 1000  # sliding window size


_state = MonitoringState()


def _ensure_log_file_exists() -> None:
    """Create the log directory and CSV file (with header) if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "text_length", "label", "score"])


def record_prediction(label: SentimentLabel, score: float, text: str) -> None:
    """Record a prediction event both in memory and on disk."""
    _state.total_requests += 1
    if label in _state.label_counts:
        _state.label_counts[label] += 1

    event = SentimentEvent(
        timestamp=datetime.utcnow(),
        text_length=len(text),
        label=label,
        score=score,
    )

    # Sliding window in memory
    _state.recent_events.append(event)
    if len(_state.recent_events) > _state.max_recent_events:
        _state.recent_events.pop(0)

    # Persistent CSV time-series log
    _ensure_log_file_exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                event.timestamp.isoformat(),
                event.text_length,
                event.label,
                f"{event.score:.4f}",
            ]
        )


def get_stats() -> Dict:
    """Return monitoring stats in a Grafana-friendly JSON structure."""
    if _state.total_requests == 0:
        label_distribution = {k: 0.0 for k in _state.label_counts.keys()}
    else:
        label_distribution = {
            label: count / _state.total_requests
            for label, count in _state.label_counts.items()
        }

    # Time-series data: recent events with timestamps
    time_series = [
        {
            "timestamp": e.timestamp.isoformat(),
            "text_length": e.text_length,
            "label": e.label,
            "score": e.score,
        }
        for e in _state.recent_events[-500:]
    ]

    return {
        "total_requests": _state.total_requests,
        "label_counts": _state.label_counts,
        "label_distribution": label_distribution,
        "time_series": time_series,
    }
