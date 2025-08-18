# tickets/agents.py
import uuid
import numpy as np
import torch
from typing import List, Tuple

from .contracts import Ticket, Tokenized, AgentOutput
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
    def __init__(self):
        ensure_loaded()
        self.tokenizer = get_tokenizer()

    def handle(self, ticket: Ticket) -> Tokenized:
        text = (ticket.subject + " " + ticket.body).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # keep simple list payload for passing across threads/executors
        return Tokenized(
            ticket_id=ticket.ticket_id,
            input_ids=enc["input_ids"].squeeze(0).to(DEVICE).tolist(),
            attention_mask=enc["attention_mask"].squeeze(0).to(DEVICE).tolist(),
        )

# ---------- Shared forward helper ----------

def _to_device(tokenized: Tokenized):
    input_ids = torch.tensor(tokenized.input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attention = torch.tensor(tokenized.attention_mask, dtype=torch.long, device=DEVICE).unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention}

def _forward(task: str, tokenized: Tokenized, multilabel: bool) -> Tuple[List[str], dict]:
    ensure_loaded()
    model  = get_model(task)
    labels = get_labels(task)
    with torch.no_grad():
        logits = model(**_to_device(tokenized)).logits.squeeze(0).detach().cpu().numpy()

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
    def handle(self, tokenized: Tokenized) -> AgentOutput:
        preds, scores = _forward(self.task, tokenized, self.multilabel)
        return AgentOutput(ticket_id=tokenized.ticket_id, task=self.task, preds=preds, scores=scores)

class DepartmentAgent:
    task = "department"
    multilabel = False
    def handle(self, tokenized: Tokenized) -> AgentOutput:
        preds, scores = _forward(self.task, tokenized, self.multilabel)
        return AgentOutput(ticket_id=tokenized.ticket_id, task=self.task, preds=preds, scores=scores)

class TypeAgent:
    task = "type"
    multilabel = False
    def handle(self, tokenized: Tokenized) -> AgentOutput:
        preds, scores = _forward(self.task, tokenized, self.multilabel)
        return AgentOutput(ticket_id=tokenized.ticket_id, task=self.task, preds=preds, scores=scores)

class PriorityAgent:
    task = "priority"
    multilabel = False
    def handle(self, tokenized: Tokenized) -> AgentOutput:
        preds, scores = _forward(self.task, tokenized, self.multilabel)
        return AgentOutput(ticket_id=tokenized.ticket_id, task=self.task, preds=preds, scores=scores)
