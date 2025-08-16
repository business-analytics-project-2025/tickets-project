# tickets/registry.py
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import BASE_MODEL, MODEL_ASSETS_DIR, FILES, DEVICE, MAX_LENGTH, USE_FAST, DO_LOWER

# Globals (lazy-loaded)
_tokenizer = None
_models: Dict[str, AutoModelForSequenceClassification] = {}
_labels: Dict[str, List[str]] = {}
_thresholds: Optional[np.ndarray] = None  # for tags only

def _assets_path(*names) -> Path:
    p = Path(MODEL_ASSETS_DIR, *names)
    if not p.exists():
        raise FileNotFoundError(f"Expected asset missing: {p}")
    return p

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, use_fast=USE_FAST, do_lower_case=DO_LOWER
        )
    return _tokenizer

def _load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Labels file must be a list[str]: {path}")
    return data

def _load_thresholds(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Thresholds must be 1-D list/array: {path}")
    return arr

def _load_model(task: str, num_labels: int, multilabel: bool, weights_path: Path):
    cfg = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        problem_type="multi_label_classification" if multilabel else "single_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

def ensure_loaded():
    """Load tokenizer, labels, thresholds (tags), and all four models once."""
    global _thresholds

    # Tokenizer
    get_tokenizer()

    # Tasks
    for task, spec in FILES.items():
        labels_path  = _assets_path(spec["labels"])
        labels       = _load_json_list(labels_path)
        _labels[task] = labels

        weights_path = _assets_path(spec["weights"])
        model        = _load_model(task, num_labels=len(labels), multilabel=spec["multilabel"], weights_path=weights_path)
        _models[task] = model

    # Per-tag thresholds (tags only)
    tags_extra = FILES["tags"]["extra"]
    if tags_extra:
        thresh_path = _assets_path(tags_extra)
        _thresholds = _load_thresholds(thresh_path)

    # Sanity: thresholds length matches labels
    if _thresholds is not None and len(_thresholds) != len(_labels["tags"]):
        raise ValueError("Length mismatch: tags thresholds vs tags labels")

def get_model(task: str):
    if task not in _models:
        ensure_loaded()
    return _models[task]

def get_labels(task: str) -> List[str]:
    if task not in _labels:
        ensure_loaded()
    return _labels[task]

def get_thresholds() -> Optional[np.ndarray]:
    if _thresholds is None:
        ensure_loaded()
    return _thresholds
