# tickets/registry.py
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Per-task model config (backbones, filenames, device, etc.)
from .config import MODEL_ASSETS_DIR, FILES, DEVICE, USE_FAST, DO_LOWER

# ---- Globals (lazy, thread-safe) ----
_tokenizers: Dict[str, AutoTokenizer] = {}  # task -> tokenizer
_models: Dict[str, AutoModelForSequenceClassification] = {}  # task -> model
_labels: Dict[str, List[str]] = {}  # task -> label list
_thresholds: Optional[np.ndarray] = None  # for tags only

_loaded: bool = False
_load_lock = threading.Lock()


def _assets_path(*names) -> Path:
    p = Path(MODEL_ASSETS_DIR, *names)
    if not p.exists():
        raise FileNotFoundError(f"Expected asset missing: {p}")
    return p


def get_tokenizer(task: str) -> AutoTokenizer:
    """Tokenizer for a given task (loaded from that task's base model)."""
    if task not in _tokenizers:
        base = FILES[task]["base_model"]
        _tokenizers[task] = AutoTokenizer.from_pretrained(
            base, use_fast=USE_FAST, do_lower_case=DO_LOWER
        )
    return _tokenizers[task]


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


# tickets/registry.py
# ... (imports and other functions are unchanged) ...

# tickets/registry.py

def _load_model(task: str, num_labels: int, multilabel: bool, weights_path: Path):
    """Instantiate the correct backbone per task and load fine-tuned head weights."""
    base = FILES[task]["base_model"]
    # 1. Create the correct model CONFIGURATION with the right number of labels.
    cfg = AutoConfig.from_pretrained(
        base,
        num_labels=num_labels,
        problem_type="multi_label_classification" if multilabel else "single_label_classification",
    )
    
    # 2. Create the model's empty ARCHITECTURE from the configuration.
    #    This creates the model scaffold with randomly initialized weights.
    model = AutoModelForSequenceClassification.from_config(cfg)

    # 3. Load the state dict from your .pt file, which contains the
    #    weights for the ENTIRE fine-tuned model.
    state = torch.load(weights_path, map_location="cpu")

    # 4. Load your fine-tuned weights into the empty scaffold.
    #    This now works because the architecture and the state dict match perfectly.
    model.load_state_dict(state, strict=True)
    
    print(f"✅ Successfully loaded full model weights for task '{task}'.")

    model.to(DEVICE)
    model.eval()
    return model


def ensure_loaded():
    """Load labels, thresholds (tags), tokenizers, and all four models once (thread-safe)."""
    global _thresholds, _loaded

    if _loaded:
        return

    with _load_lock:
        if _loaded:
            return

        # 1) Labels and tokenizers
        for task, spec in FILES.items():
            labels_path = _assets_path(spec["labels"])
            _labels[task] = _load_json_list(labels_path)
            # touch tokenizer so it is ready for parallel calls
            get_tokenizer(task)

        # 2) Models
        for task, spec in FILES.items():
            weights_path = _assets_path(spec["weights"])
            _models[task] = _load_model(
                task,
                num_labels=len(_labels[task]),
                multilabel=spec["multilabel"],
                weights_path=weights_path,
            )

        # 3) Per-tag thresholds (tags only)
        tags_extra = FILES["tags"].get("extra")
        if tags_extra:
            thresh_path = _assets_path(tags_extra)
            _thresholds = _load_thresholds(thresh_path)

        # 4) Sanity checks
        if _thresholds is not None and len(_thresholds) != len(_labels["tags"]):
            raise ValueError("Length mismatch: tags thresholds vs tags labels")

        _loaded = True


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
