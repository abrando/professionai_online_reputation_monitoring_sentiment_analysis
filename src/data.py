# src/data.py
"""
Data utilities for future retraining.

This module defines how new labeled data should be stored and loaded.
The idea is that future retraining will consume CSV files from data/new.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

NEW_DATA_DIR = Path("data/new")


def list_new_data_files() -> List[Path]:
    """List CSV files containing newly labeled data for potential retraining."""
    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(NEW_DATA_DIR.glob("*.csv"))


def load_labeled_data(path: Path) -> Tuple[pd.Series, pd.Series]:
    """Load a labeled dataset from a CSV file.

    Expected columns:
      - text: the social media text
      - label: integer label (0=negative, 1=neutral, 2=positive)
    """
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"File {path} must contain 'text' and 'label' columns")

    return df["text"], df["label"]
