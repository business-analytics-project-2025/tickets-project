import os
from pathlib import Path

MODEL_ASSETS_DIR = Path.cwd() / "models_weights"

MAX_LENGTH  = 256
USE_FAST    = False
DO_LOWER    = True
DEVICE      = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") or os.environ.get("FORCE_CUDA") else "cpu"

FILES = {
    "tags": {
        "base_model": "microsoft/deberta-v3-large",
        "weights": "pretrained_tags_model_weights.pt",
        "labels":  "tags_labels.json",
        "extra":   "tags_thresholds.json",
        "multilabel": True,
    },
    "department": {
        "base_model": "bert-base-cased",
        "weights": "pretrained_department_model_weights.pt",
        "labels":  "department_labels.json",
        "extra":   None,
        "multilabel": False,
    },
    "type": {
        "base_model": "distilbert-base-uncased",
        "weights": "pretrained_type_model_weights.pt",
        "labels":  "type_labels.json",
        "extra":   None,
        "multilabel": False,
    },
    "priority": {
        "base_model": "distilbert-base-uncased",
        "weights": "pretrained_priority_model_weights.pt",
        "labels":  "priority_labels.json",
        "extra":   None,
        "multilabel": False,
    },
}

PRIORITY_TO_CLICKUP = {"High": 2, "Medium": 3, "Low": 4}

TYPE_FIELD_ID = "660e1b3b-ec41-40a6-9863-979e44951c70"
DEPT_FIELD_ID = "90647e3f-2209-4d5c-be47-7780a883ac28"