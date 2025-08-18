# tickets/inference.py
from typing import Dict, List
import numpy as np
import torch

from .registry import get_tokenizer, get_model, get_labels, get_thresholds, ensure_loaded
from .config import MAX_LENGTH, DEVICE

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))

def _tokenize_once(subject: str, body: str):
    tok = get_tokenizer()
    text = f"{(subject or '').strip()} {(body or '').strip()}".strip()
    enc = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}, text

def _forward(task: str, tokenized_inputs, multilabel: bool):
    model = get_model(task)
    with torch.no_grad():
        logits = model(**tokenized_inputs).logits.squeeze(0).detach().cpu().numpy()
    labels = get_labels(task)

    if multilabel:
        probs = _SIGMOID(logits)
        thresholds = get_thresholds()
        if thresholds is None:
            # default 0.5 if thresholds missing
            pred_mask = probs > 0.5
        else:
            pred_mask = probs > thresholds
        preds = [labels[i] for i, on in enumerate(pred_mask) if on]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    else:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        idx = int(np.argmax(probs))
        preds = [labels[idx]]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return preds, scores

def predict_all(subject: str, body: str) -> Dict:
    """Single entrypoint used by agents/UI/MCP: tokenize once, run all models, return dict."""
    ensure_loaded()
    tokenized, _ = _tokenize_once(subject, body)

    tags_preds, tags_scores         = _forward("tags",        tokenized, multilabel=True)
    dept_preds, dept_scores         = _forward("department",  tokenized, multilabel=False)
    type_preds, type_scores         = _forward("type",        tokenized, multilabel=False)
    priority_preds, priority_scores = _forward("priority",    tokenized, multilabel=False)

    return {
        "tags": tags_preds,
        "department": dept_preds[0] if dept_preds else "",
        "type": type_preds[0] if type_preds else "",
        "priority": priority_preds[0] if priority_preds else "",
        "confidences": {
            "tags": tags_scores,
            "department": dept_scores,
            "type": type_scores,
            "priority": priority_scores,
        },
        # Optionally: versions or metadata can be added later
    }
