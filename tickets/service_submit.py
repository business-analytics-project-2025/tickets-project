# tickets/service_submit.py
"""
Service entrypoints that can be called by an MCP server or any Python client.

submit_ticket(subject, body) -> dict
  - fail-fast: if any model/agent fails, returns {ok: False, reason: "..."} and does NOT create a task
  - duplicate block: hash(subject||body); return {ok: False, reason: "duplicate"} if seen within TTL
  - on success: creates ClickUp task, sets priority, type, department, tags; returns {ok: True, task_id, task_url}
"""

from typing import Dict

from .orchestrator import predict_all as orchestrator_predict_all
from .duplicate_check import dedupe, remember
from .clickup_client import create_task, add_tags, set_dropdown_value
from .config import PRIORITY_TO_CLICKUP

# ClickUp custom field IDs you provided
TYPE_FIELD_ID  = "660e1b3b-ec41-40a6-9863-979e44951c70"
DEPT_FIELD_ID  = "90647e3f-2209-4d5c-be47-7780a883ac28"

def submit_ticket(subject: str, body: str) -> Dict:
    """
    Orchestrate prediction -> duplicate check -> ClickUp creation (all-or-nothing).

    Returns:
      {"ok": True, "task_id": "...", "task_url": "..."} on success
      {"ok": False, "reason": "duplicate" | "prediction_failed:..." | "clickup_failed:..."} on failure
    """
    # Duplicate block
    is_dup, h = dedupe(subject, body)
    if is_dup:
        return {"ok": False, "reason": "duplicate"}

    # Predict (fail fast, no partials allowed)
    try:
        pred = orchestrator_predict_all(subject, body)
    except Exception as e:
        return {"ok": False, "reason": f"prediction_failed: {e}"}

    # Map priority to ClickUp numeric
    priority_num = PRIORITY_TO_CLICKUP.get(pred.get("priority", ""), 3)

    # Create task + set fields/tags (auto-create options and tags as needed)
    try:
        subject_safe = (subject or "").strip() or "(no subject)"
        body_safe    = (body or "").strip()

        task = create_task(
            name=subject_safe,
            description=body_safe,
            priority_num=priority_num
        )
        task_id = task["id"]

        # Custom dropdowns: Type & Department (auto-add option if missing)
        type_val = pred.get("type") or ""
        dept_val = pred.get("department") or ""
        if type_val:
            set_dropdown_value(task_id, TYPE_FIELD_ID, type_val)
        if dept_val:
            set_dropdown_value(task_id, DEPT_FIELD_ID, dept_val)

        # Tags (multi-label): auto-create if missing
        tags = pred.get("tags") or []
        if tags:
            add_tags(task_id, tags)

    except Exception as e:
        return {"ok": False, "reason": f"clickup_failed: {e}"}

    # Remember hash after success to block duplicates next time
    remember(h)
    return {"ok": True, "task_id": task_id, "task_url": task.get("url", "")}
