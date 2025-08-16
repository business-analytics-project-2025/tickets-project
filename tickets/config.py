import os
from pathlib import Path

# --- Assets (your path) ---
MODEL_ASSETS_DIR = Path(r"C:\Users\chria\Documents\Μεταπτυχιακό\Deep Learning\tickets-project\models")

# --- HF model & preprocessing (match your training) ---
BASE_MODEL  = "microsoft/deberta-v3-large"
MAX_LENGTH  = 256
USE_FAST    = False
DO_LOWER    = True
DEVICE      = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") or os.environ.get("FORCE_CUDA") else "cpu"

# --- Filenames (conventions you gave) ---
FILES = {
    "tags": {
        "weights": "pretrained_tags_model_weights.pt",
        "labels":  "tags_labels.json",
        "extra":   "tags_thresholds.json",  # per-tag thresholds
        "multilabel": True,
    },
    "department": {
        "weights": "pretrained_department_model_weights.pt",
        "labels":  "department_labels.json",
        "extra":   None,
        "multilabel": False,
    },
    "type": {
        "weights": "pretrained_type_model_weights.pt",
        "labels":  "type_labels.json",
        "extra":   None,
        "multilabel": False,
    },
    "priority": {
        "weights": "pretrained_priority_model_weights.pt",
        "labels":  "priority_labels.json",
        "extra":   None,
        "multilabel": False,
    },
}

# Optional: priority mapping for later (ClickUp)
PRIORITY_TO_CLICKUP = {"High": 2, "Medium": 3, "Low": 4}
