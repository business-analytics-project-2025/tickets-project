# tickets/service_submit.py
"""
Service entrypoints callable by the MCP server (or tests).

submit_ticket(subject, body) -> dict
  - Fail-fast: if ANY classifier agent fails, returns {ok: False, reason: "..."} and DOES NOT create a task
  - Duplicate block: hash(subject || body); returns {ok: False, reason: "duplicate"} if seen recently
  - On success: creates ClickUp task, sets priority, type, department, tags; returns {ok: True, task_id, task_url}
"""

from typing import Dict

from .orchestrator import predict_all as orchestrator_predict_all
from .contracts import CombinedPredictionError
from .duplicate_check import dedupe, remember
from .clickup_client import create_task, add_tags, set_dropdown_value
from .config import PRIORITY_TO_CLICKUP

# ClickUp custom field IDs for your List:
TYPE_FIELD_ID = "660e1b3b-ec41-40a6-9863-979e44951c70"
DEPT_FIELD_ID = "90647e3f-2209-4d5c-be47-7780a883ac28"

def submit_ticket(subject: str, body: str) -> Dict:
    """
    Orchestrate prediction -> duplicate check -> ClickUp creation (all-or-nothing).

    Returns:
      {"ok": True, "task_id": "...", "task_url": "..."} on success
      {"ok": False, "reason": "duplicate" | "prediction_failed:..." | "clickup_failed:..."} on failure
    """
    # 0) Duplicate block
    is_dup, h = dedupe(subject, body)
    if is_dup:
        return {"ok": False, "reason": "duplicate"}

    # 1) Predict (fail-fast; surface the failing agent when possible)
    try:
        pred = orchestrator_predict_all(subject, body)
    except CombinedPredictionError as e:
        # e.detail may include {"agent": "...", "cause": "..."}
        return {"ok": False, "reason": f"prediction_failed: {e.detail or str(e)}"}
    except Exception as e:
        return {"ok": False, "reason": f"prediction_failed: {e}"}

    # 2) Map priority string -> ClickUp numeric (default Medium=3)
    priority_str = (pred.get("priority") or "").strip()
    priority_num = PRIORITY_TO_CLICKUP.get(priority_str, 3)

    # 3) Create task + set custom fields + tags
    try:
        name = (subject or "").strip() or "(no subject)"
        desc = (body or "").strip()

        task = create_task(name=name, description=desc, priority_num=priority_num)
        task_id = task["id"]

        # Type (dropdown) — auto-add option if missing
        tval = (pred.get("type") or "").strip()
        if tval:
            set_dropdown_value(task_id, TYPE_FIELD_ID, tval)

        # Department (dropdown) — auto-add option if missing
        dval = (pred.get("department") or "").strip()
        if dval:
            set_dropdown_value(task_id, DEPT_FIELD_ID, dval)

        # Tags (multi-label) — auto-create if missing
        tags = pred.get("tags") or []
        if tags:
            add_tags(task_id, tags)

    except Exception as e:
        return {"ok": False, "reason": f"clickup_failed: {e}"}

    # 4) Remember hash to block duplicates next time
    remember(h)

    return {"ok": True, "task_id": task_id, "task_url": task.get("url", "")}
