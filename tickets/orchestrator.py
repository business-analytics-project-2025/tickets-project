# tickets/orchestrator.py
import os
import asyncio
from typing import Dict

from .contracts import FinalPrediction, CombinedPredictionError
# This import has been updated from .agents to .ml_models
from .ml_models import (
    IntakeAgent,
    PreprocessAgent,
    TagsAgent,
    DepartmentAgent,
    TypeAgent,
    PriorityAgent,
)
from .registry import ensure_loaded

# ---- Tunables ----
# DeBERTa-v3-large on CPU can exceed 8s; make it configurable.
AGENT_TIMEOUT_SEC = float(os.getenv("AGENT_TIMEOUT_SEC", "30"))
RETRY_ONCE = True


class Orchestrator:
    def __init__(self):
        self.intake = IntakeAgent()
        self.pre = PreprocessAgent()
        self.tags = TagsAgent()
        self.dept = DepartmentAgent()
        self.typ = TypeAgent()
        self.prio = PriorityAgent()

    async def _call_with_retry(self, fn, *args):
        async def _once():
            # Run blocking model call in a thread to avoid blocking the event loop
            return await asyncio.to_thread(fn, *args)

        try:
            return await asyncio.wait_for(_once(), timeout=AGENT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            # Retry won't help if we already hit the timeout; fail fast with clear message
            raise RuntimeError("timeout")
        except Exception:
            if not RETRY_ONCE:
                raise
            # One retry (fresh thread)
            return await asyncio.wait_for(_once(), timeout=AGENT_TIMEOUT_SEC)

    async def predict(self, subject: str, body: str) -> FinalPrediction:
        # Intake + light preprocess (no tokenization here; each agent tokenizes itself)
        ticket = self.intake.handle(subject, body)
        ticket = self.pre.handle(ticket)

        # Ensure all tokenizers/models are fully loaded before parallel calls
        ensure_loaded()

        # Fan-out to specialist agents
        tasks = {
            "tags": self._call_with_retry(self.tags.handle, ticket),
            "department": self._call_with_retry(self.dept.handle, ticket),
            "type": self._call_with_retry(self.typ.handle, ticket),
            "priority": self._call_with_retry(self.prio.handle, ticket),
        }

        # Gather results; do not raise immediately so we can attach which agent failed
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Fail-fast with clear agent + cause
        for (name, _), res in zip(tasks.items(), results):
            if isinstance(res, Exception):
                raise CombinedPredictionError(
                    "agent_failed",
                    {"agent": name, "cause": str(res)},
                )

        a_tags, a_dept, a_type, a_prio = results

        return FinalPrediction(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            body=ticket.body,
            tags=a_tags.preds,
            department=a_dept.preds[0] if a_dept.preds else "",
            type=a_type.preds[0] if a_type.preds else "",
            priority=a_prio.preds[0] if a_prio.preds else "",
            confidences={
                "tags": a_tags.scores,
                "department": a_dept.scores,
                "type": a_type.scores,
                "priority": a_prio.scores,
            },
        )


def predict_all(subject: str, body: str) -> Dict:
    """
    Synchronous convenience wrapper that runs the orchestrator and
    returns a plain dict for callers (API service, tests, etc.).
    """
    async def _run():
        orch = Orchestrator()
        out = await orch.predict(subject, body)
        return {
            "ticket_id": out.ticket_id,
            "subject": out.subject,
            "body": out.body,
            "tags": out.tags,
            "department": out.department,
            "type": out.type,
            "priority": out.priority,
            "confidences": out.confidences,
        }

    return asyncio.run(_run())