"""Preprocessing utilities for the scheduling model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import json
import re


@dataclass
class PreprocessedData:
    """Container for preprocessed schedule configuration."""

    config: Dict[str, Any]
    teacher_ids: Dict[str, int] = field(default_factory=dict)
    student_ids: Dict[str, int] = field(default_factory=dict)
    cabinet_ids: Dict[str, int] = field(default_factory=dict)
    subject_ids: Dict[str, int] = field(default_factory=dict)


def load_config(path: str = "schedule-config.json") -> Dict[str, Any]:
    """Load raw configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"//.*?$|#.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL)
    return json.loads(text)


def filter_domains(cfg: Dict[str, Any]) -> None:
    """Remove impossible scheduling options using availability rules."""
    # TODO: implement domain filtering based on teacher and room availability
    pass


def map_ids(cfg: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Map teachers, students, cabinets and subjects to contiguous integer IDs."""
    # TODO: implement deterministic ID remapping
    return {}, {}, {}, {}


def preprocess(path: str = "schedule-config.json") -> PreprocessedData:
    """Perform basic preprocessing and return structured data."""
    cfg = load_config(path)
    filter_domains(cfg)
    tids, sids, cids, subids = map_ids(cfg)
    return PreprocessedData(
        config=cfg,
        teacher_ids=tids,
        student_ids=sids,
        cabinet_ids=cids,
        subject_ids=subids,
    )


def save_preprocessed(data: PreprocessedData, path: str) -> None:
    """Serialise preprocessed data to a JSON file."""
    # TODO: store as pickle or JSON for reuse
    pass


def load_preprocessed(path: str) -> PreprocessedData:
    """Load preprocessed data previously saved."""
    # TODO: read the cached file and recreate PreprocessedData
    return PreprocessedData(config={})
