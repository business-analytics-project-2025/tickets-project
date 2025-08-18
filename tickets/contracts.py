# tickets/contracts.py
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Ticket:
    ticket_id: str
    subject: str
    body: str

@dataclass
class Tokenized:
    ticket_id: str
    input_ids: List[int]
    attention_mask: List[int]

@dataclass
class AgentOutput:
    ticket_id: str
    task: str                      # "tags" | "department" | "type" | "priority"
    preds: List[str]               # multi-label: list; single-label: [label]
    scores: Dict[str, float]       # label -> probability/logit
    model_version: str = "v1"      # placeholder; wire real versions later

@dataclass
class FinalPrediction:
    ticket_id: str
    subject: str
    body: str
    tags: List[str]
    department: str
    type: str
    priority: str
    confidences: Dict[str, Dict[str, float]]   # task -> (label -> score)

class CombinedPredictionError(RuntimeError):
    """Raised when any agent fails; no partials allowed."""
    def __init__(self, message: str, detail: Optional[Dict]=None):
        super().__init__(message)
        self.detail = detail or {}
