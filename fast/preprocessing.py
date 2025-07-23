"""Preprocessing utilities for the scheduling model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


@dataclass
class PreprocessedData:
    """Container for preprocessed schedule configuration."""

    config: Dict[str, Any]


def load_config(path: str = "schedule-config.json") -> Dict[str, Any]:
    """Load raw configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def preprocess(path: str = "schedule-config.json") -> PreprocessedData:
    """Perform basic preprocessing and return structured data."""
    cfg = load_config(path)
    # TODO: add domain filtering and ID remapping
    return PreprocessedData(config=cfg)
