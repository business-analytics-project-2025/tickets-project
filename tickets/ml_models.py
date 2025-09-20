import uuid
import numpy as np
import torch
from typing import List, Tuple

from .contracts import Ticket, AgentOutput
from .config import MAX_LENGTH, DEVICE
from .registry import get_tokenizer, get_model, get_labels, get_thresholds, ensure_loaded

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))

# ---------- Intake & Preprocess ----------

class IntakeAgent:
    def handle(self, subject: str, body: str) -> Ticket:
        s = (subject or "").strip()
        b = (body or "").strip()
        if not s and not b:
            raise ValueError("Empty ticket: both subject and body are empty.")
        return Ticket(ticket_id=str(uuid.uuid4()), subject=s, body=b)

class PreprocessAgent:
    """With mixed backbones, we don't tokenize here anymore.
    We just normalize and pass the Ticket through.
    """
    def __init__(self):
        ensure_loaded()

    def handle(self, ticket: Ticket) -> Ticket:
        ticket.subject = ticket.subject.strip()
        ticket.body = ticket.body.strip()
        return ticket

# ---------- Per-task forward helpers ----------

def _tokenize_for_task(task: str, text: str):
    tok = get_tokenizer(task)
    enc = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}

def _forward(task: str, ticket: Ticket, multilabel: bool) -> Tuple[List[str], dict]:
    ensure_loaded()
    model  = get_model(task)
    labels = get_labels(task)

    text = (ticket.subject + " " + ticket.body).strip()
    inputs = _tokenize_for_task(task, text)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0).detach().cpu().numpy()

    if multilabel:
        probs = _SIGMOID(logits)
        thresholds = get_thresholds()
        pred_mask = probs > (thresholds if thresholds is not None else 0.5)
        preds  = [labels[i] for i, on in enumerate(pred_mask) if on]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    else:
        probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        idx    = int(np.argmax(probs))
        preds  = [labels[idx]]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return preds, scores

# ---------- Specialist classifier agents ----------

class TagsAgent:
    task = "tags"
    multilabel = True
    def handle(self, ticket: Ticket) -> AgentOutput:
        preds, scores = _forward(self.task, ticket, self.multilabel)
        return AgentOutput(ticket_id=ticket.ticket_id, task=self.task, preds=preds, scores=scores)

class DepartmentAgent:
    task = "department"
    multilabel = False
    def handle(self, ticket: Ticket) -> AgentOutput:
        preds, scores = _forward(self.task, ticket, self.multilabel)
        return AgentOutput(ticket_id=ticket.ticket_id, task=self.task, preds=preds, scores=scores)

class TypeAgent:
    task = "type"
    multilabel = False
    def handle(self, ticket: Ticket) -> AgentOutput:
        preds, scores = _forward(self.task, ticket, self.multilabel)
        return AgentOutput(ticket_id=ticket.ticket_id, task=self.task, preds=preds, scores=scores)

class PriorityAgent:
    task = "priority"
    multilabel = False
    def handle(self, ticket: Ticket) -> AgentOutput:
        preds, scores = _forward(self.task, ticket, self.multilabel)
        return AgentOutput(ticket_id=ticket.ticket_id, task=self.task, preds=preds, scores=scores)
