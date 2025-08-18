# tickets/orchestrator.py
import asyncio
from typing import Dict

from .contracts import FinalPrediction, CombinedPredictionError
from .agents import IntakeAgent, PreprocessAgent, TagsAgent, DepartmentAgent, TypeAgent, PriorityAgent

# Tunables
AGENT_TIMEOUT_SEC = 8.0
RETRY_ONCE = True

class Orchestrator:
    def __init__(self):
        self.intake = IntakeAgent()
        self.pre = PreprocessAgent()
        self.tags = TagsAgent()
        self.dept = DepartmentAgent()
        self.typ  = TypeAgent()
        self.prio = PriorityAgent()

    async def _call_with_retry(self, fn, *args):
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=AGENT_TIMEOUT_SEC)
        except Exception as e:
            if not RETRY_ONCE:
                raise
            # one retry
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=AGENT_TIMEOUT_SEC)

    async def predict(self, subject: str, body: str) -> FinalPrediction:
        # Intake + preprocess
        ticket = self.intake.handle(subject, body)
        tokenized = self.pre.handle(ticket)

        # Fan-out to specialist agents
        tasks = [
            self._call_with_retry(self.tags.handle, tokenized),
            self._call_with_retry(self.dept.handle, tokenized),
            self._call_with_retry(self.typ.handle,  tokenized),
            self._call_with_retry(self.prio.handle, tokenized),
        ]

        # Fail-fast: if any fails, cancel the rest and raise CombinedPredictionError
        try:
            a_tags, a_dept, a_type, a_prio = await asyncio.gather(*tasks)
        except Exception as e:
            for t in tasks:
                if isinstance(t, asyncio.Task) and not t.done():
                    t.cancel()
            raise CombinedPredictionError("Prediction failed in at least one agent", {"cause": str(e)})

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

# Convenience wrapper so callers don’t have to write asyncio themselves
def predict_all(subject: str, body: str) -> Dict:
    """Sync wrapper that runs the orchestrator and returns a plain dict."""
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
