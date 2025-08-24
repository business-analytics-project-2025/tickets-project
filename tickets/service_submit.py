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
from .clickup_client import (
    create_task,
    add_tags,
    set_dropdown_value,
    append_tags_note,
    append_field_note,
)
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
        return {"ok": False, "reason": f"prediction_failed: {e.detail or str(e)}"}
    except Exception as e:
        return {"ok": False, "reason": f"prediction_failed: {e}"}

    # 2) Map priority string -> ClickUp numeric (default Medium=3)
    priority_str = (pred.get("priority") or "").strip()
    priority_num = PRIORITY_TO_CLICKUP.get(priority_str, 3)

    # 3) Create task
    try:
        name = (subject or "").strip() or "(no subject)"
        desc = (body or "").strip()
        task = create_task(name=name, description=desc, priority_num=priority_num)
        task_id = task["id"]
    except Exception as e:
        return {"ok": False, "reason": f"clickup_failed: create_task: {e}"}

    # 4) Set dropdowns (strict match to existing options; append note when fuzzy)
    try:
        tval = (pred.get("type") or "").strip()
        if tval:
            _, exact, chosen_name, _ = set_dropdown_value(task_id, TYPE_FIELD_ID, tval)
            if not exact:
                append_field_note(task_id, "Type", tval, chosen_name)
    except Exception as e:
        return {"ok": False, "reason": f"clickup_failed: type_set: {e}"}

    try:
        dval = (pred.get("department") or "").strip()
        if dval:
            _, exact, chosen_name, _ = set_dropdown_value(task_id, DEPT_FIELD_ID, dval)
            if not exact:
                append_field_note(task_id, "Department", dval, chosen_name)
    except Exception as e:
        return {"ok": False, "reason": f"clickup_failed: department_set: {e}"}

    # 5) Tags (best-effort; record in description if attach fails)
    try:
        tags = pred.get("tags") or []
        if tags:
            failed = add_tags(task_id, tags)
            if failed:
                append_tags_note(task_id, failed)
    except Exception as e:
        # Don't fail the entire submission because of tags; just note it.
        # If you want strict behavior, change this to: return {"ok": False, "reason": f"clickup_failed: tags: {e}"}
        append_tags_note(task_id, tags or [])
        # continue

    # 6) Remember hash to block duplicates next time
    remember(h)

    return {"ok": True, "task_id": task_id, "task_url": task.get("url", "")}
