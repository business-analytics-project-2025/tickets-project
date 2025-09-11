# agent_runner.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

# --- LLM (local via Ollama) ---
from langchain_community.chat_models import ChatOllama

# --- Light cleaner tool ---
from text_clean import clean_subject_body

# --- Pipeline + ClickUp imports (support both flat and 'tickets.' package layouts) ---
try:
    # If your files live under a 'tickets/' package
    from tickets.orchestrator import predict_all as pipeline_predict_all
    from tickets.clickup_client import (
        create_task,
        add_tags,
        set_dropdown_value,
        append_tags_note,
        append_field_note,
        ClickUpHTTPError,
    )
    from tickets.config import PRIORITY_TO_CLICKUP
    from tickets.service_submit import TYPE_FIELD_ID, DEPT_FIELD_ID
except Exception:
    # If your files are flat at repo root
    from tickets.orchestrator import predict_all as pipeline_predict_all
    from tickets.clickup_client import (
        create_task,
        add_tags,
        set_dropdown_value,
        append_tags_note,
        append_field_note,
        ClickUpHTTPError,
    )
    from tickets.config import PRIORITY_TO_CLICKUP
    from tickets.service_submit import TYPE_FIELD_ID, DEPT_FIELD_ID


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _obs_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s or "")
        return json.loads(m.group(0)) if m else {"ok": False, "reason": "non_json_observation"}


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
def tool_clean_text(inp: str) -> str:
    """Input: {"subject": str, "body": str} -> {"ok": true, "subject": str, "body": str}"""
    try:
        data = json.loads(inp or "{}")
        s, b = clean_subject_body(data.get("subject", ""), data.get("body", ""))
        return _as_json({"ok": True, "subject": s, "body": b})
    except Exception as e:
        return _as_json({"ok": False, "reason": f"clean_failed: {e}"})


def tool_predict_pipeline(inp: str) -> str:
    """
    Input: {"subject": str, "body": str}
    Return: {"ok": true, "pred": {priority, type, department, tags}} or {"ok": false, "reason": "..."}
    """
    try:
        data = json.loads(inp or "{}")
        subject = (data.get("subject") or "").strip()
        body = (data.get("body") or "").strip()
        if not subject and not body:
            return _as_json({"ok": False, "reason": "prediction_failed: Empty ticket after preprocessing"})
        pred = pipeline_predict_all(subject, body)
        return _as_json({"ok": True, "pred": pred})
    except Exception as e:
        return _as_json({"ok": False, "reason": f"prediction_failed: {e}"})


def tool_create_clickup_task(inp: str) -> str:
    """
    Input:  {"subject": str, "body": str, "pred": {...}}
    Output: {"ok": true, "task_id": "...", "task_url": "..."} OR {"ok": false, "reason": "..."}
    """
    import json
    try:
        data = json.loads(inp or "{}")
        subject = (data.get("subject") or "").strip() or "(no subject)"
        body    = (data.get("body") or "").strip()
        pred    = data.get("pred") or {}

        # Priority mapping (1=urgent, 2=high, 3=normal, 4=low)
        priority_str = (pred.get("priority") or "").strip()
        priority_num = PRIORITY_TO_CLICKUP.get(priority_str, 3)

        # --- Create the task (tolerant to multiple return shapes) ---
        task_raw = create_task(name=subject, description=body, priority_num=priority_num)

        # Normalize task payload -> task_obj dict with at least {"id": "...", "url": "...?"}
        if isinstance(task_raw, dict):
            task_obj = task_raw.get("task") if "task" in task_raw else task_raw
        elif isinstance(task_raw, str):
            # Could be a plain id or a JSON string
            try:
                maybe = json.loads(task_raw)
                task_obj = maybe.get("task") if isinstance(maybe, dict) and "task" in maybe else maybe
            except Exception:
                task_obj = {"id": task_raw}
        else:
            task_obj = {"raw": str(task_raw)}

        task_id = str(task_obj.get("id")) if isinstance(task_obj, dict) else None
        if not task_id:
            return _as_json({
                "ok": False,
                "reason": f"clickup_failed: create_task_return_shape: {type(task_raw).__name__}"
            })

        # Synthesize task_url if not provided by API
        task_url = ""
        if isinstance(task_obj, dict):
            task_url = str(task_obj.get("url") or "")
        if not task_url:
            task_url = f"https://app.clickup.com/t/{task_id}"

        # --- TYPE dropdown (tolerant: no-op is success; note if fuzzy) ---
        tval = (pred.get("type") or "").strip()
        if tval:
            try:
                _, exact, chosen, _ = set_dropdown_value(task_id, TYPE_FIELD_ID, tval)
                if not exact:
                    append_field_note(task_id, "Type", tval, chosen)
            except ClickUpHTTPError as e:
                return _as_json({"ok": False, "reason": f"clickup_failed: type_set: {e}"})

        # --- DEPARTMENT dropdown (tolerant: no-op is success; note if fuzzy) ---
        dval = (pred.get("department") or "").strip()
        if dval:
            try:
                _, exact, chosen, _ = set_dropdown_value(task_id, DEPT_FIELD_ID, dval)
                if not exact:
                    append_field_note(task_id, "Department", dval, chosen)
            except ClickUpHTTPError as e:
                return _as_json({"ok": False, "reason": f"clickup_failed: department_set: {e}"})

        # --- TAGS (best effort; append note if any failures) ---
        tags = pred.get("tags") or []
        try:
            failed = add_tags(task_id, tags)
            if failed:
                append_tags_note(task_id, failed)
        except Exception:
            # If tag attach fails wholesale, still leave a note with requested tags
            append_tags_note(task_id, tags or [])

        return _as_json({"ok": True, "task_id": task_id, "task_url": task_url})

    except Exception as e:
        return _as_json({"ok": False, "reason": f"clickup_failed: create_or_set: {e}"})



# ----------------------------------------------------------------------
# Minimal custom ReAct loop (agentic, robust)
# ----------------------------------------------------------------------
ALLOWED_TOOLS = ["clean_text", "predict_pipeline", "create_clickup_task"]

SYSTEM_PROMPT = (
    "You are a disciplined ReAct agent that MUST use tools.\n"
    "Protocol (exactly):\n"
    "Thought: describe next step\n"
    "Action: one of [clean_text, predict_pipeline, create_clickup_task]\n"
    "Action Input: a single JSON object (no code fences)\n"
    "Observation: the tool's JSON result\n"
    "…repeat until done… then:\n"
    "Final Answer: a single JSON object only.\n\n"
    "Hard rules:\n"
    "1) Always call clean_text FIRST.\n"
    "2) If any Observation contains {\"ok\": false, ...}, STOP and output that JSON as the Final Answer.\n"
    "3) After clean_text, call predict_pipeline, then create_clickup_task, and STOP.\n"
    "4) The Final Answer must be ONLY one JSON object (no extra text, no code fences).\n"
    "5) Never invent tool names; use only [clean_text, predict_pipeline, create_clickup_task].\n"
    "6) Always provide valid JSON in Action Input (no comments, no trailing commas).\n\n"
    "Respond in this EXACT template each step (no extra lines):\n"
    "Thought: <your short thought>\n"
    "Action: <tool name>\n"
    "Action Input: {\"key\": \"value\"}\n\n"
    "Mini example (success):\n"
    "Thought: I will clean the text.\n"
    "Action: clean_text\n"
    "Action Input: {\"subject\":\"s\",\"body\":\"b\"}\n"
    "Observation: {\"ok\": true, \"subject\": \"s2\", \"body\": \"b2\"}\n"
    "Thought: I will run the ML pipeline to get predictions.\n"
    "Action: predict_pipeline\n"
    "Action Input: {\"subject\":\"s2\",\"body\":\"b2\"}\n"
    "Observation: {\"ok\": true, \"pred\": {\"priority\": \"Normal\", \"type\": \"Tech Support\", \"department\": \"IT\", \"tags\": [\"Network\"]}}\n"
    "Thought: I will create the ClickUp task and set fields.\n"
    "Action: create_clickup_task\n"
    "Action Input: {\"subject\":\"s2\",\"body\":\"b2\",\"pred\": {\"priority\": \"Normal\", \"type\": \"Tech Support\", \"department\": \"IT\", \"tags\": [\"Network\"]}}\n"
    "Observation: {\"ok\": true, \"task_id\": \"123\", \"task_url\": \"https://...\"}\n"
    "Final Answer: {\"ok\": true, \"task_id\": \"123\", \"task_url\": \"https://...\"}\n"
)


def _render_prompt(input_json: str, scratchpad: str) -> str:
    return SYSTEM_PROMPT + "\n" + "Input ticket JSON:\n" + input_json + "\n\n" + (scratchpad or "")


_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.+?)\s*"
    r"Action:\s*(?P<tool>\w+)\s*"
    r"Action Input:\s*(?P<input>\{[\s\S]*\})",
    re.IGNORECASE | re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    # Remove ```json ... ``` or ``` ... ``` fences if the model adds them
    return re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)


def _parse_action(text: str) -> Tuple[str, Dict[str, Any], str]:
    """Extract (tool, payload_dict, thought) from model text. Raises on failure."""
    txt = _strip_code_fences(text).strip()
    m = _ACTION_RE.search(txt)
    if not m:
        raise ValueError("parse_error: missing 'Action' or 'Action Input' with JSON")
    tool = m.group("tool").strip()
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"invalid_tool: {tool}")
    raw = m.group("input")
    try:
        payload = json.loads(raw)
    except Exception as e:
        raise ValueError(f"invalid_json_for_action_input: {e}")
    thought = m.group("thought").strip()
    return tool, payload, thought


def _call_tool(tool: str, payload: Dict[str, Any]) -> str:
    if tool == "clean_text":
        return tool_clean_text(json.dumps(payload, ensure_ascii=False))
    if tool == "predict_pipeline":
        return tool_predict_pipeline(json.dumps(payload, ensure_ascii=False))
    if tool == "create_clickup_task":
        return tool_create_clickup_task(json.dumps(payload, ensure_ascii=False))
    return json.dumps({"ok": False, "reason": f"invalid_tool: {tool}"}, ensure_ascii=False)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def run_agent(subject: str, body: str) -> Dict[str, Any]:
    """
    Returns one dict:
      {"ok": true, "task_id": "...", "task_url": "..."} OR {"ok": false, "reason": "..."}.
    """
    llm = ChatOllama(model="llama3", temperature=0.0, num_ctx=4096, num_predict=512)
    input_json = json.dumps({"subject": subject, "body": body}, ensure_ascii=False)

    scratch = ""
    last_clean = None
    last_pred = None

    for step in range(1, 4):
        # Ask the LLM for the next Thought/Action
        resp = llm.invoke(_render_prompt(input_json, scratch))
        text = getattr(resp, "content", "") if resp else ""

        # Try to parse the next Action
        try:
            tool, payload, thought = _parse_action(text)
        except Exception:
            # one focused retry with explicit template
            nudge = (
                f"{scratch}"
                "Your previous message did not include a valid Action / Action Input JSON.\n"
                "Respond in EXACTLY this format (no code fences):\n"
                "Thought: <your short thought>\n"
                f"Action: <one of {ALLOWED_TOOLS}>\n"
                "Action Input: {\"key\":\"value\"}\n"
            )
            resp2 = llm.invoke(_render_prompt(input_json, nudge))
            text2 = getattr(resp2, "content", "") if resp2 else ""
            try:
                tool, payload, thought = _parse_action(text2)
            except Exception:
                # Enforce correct tool for this step with best-known payload
                if step == 1:
                    tool = "clean_text"
                    payload = json.loads(input_json)
                    thought = "I will clean the text."
                elif step == 2:
                    tool = "predict_pipeline"
                    base = json.loads(input_json)
                    subj_clean = (last_clean or {}).get("subject")
                    body_clean = (last_clean or {}).get("body")
                    subj = (subj_clean if (subj_clean or "").strip() else base["subject"]) or ""
                    bod = (body_clean if (body_clean or "").strip() else base["body"]) or ""
                    subj = subj.strip() or base["subject"].strip()
                    bod = bod.strip() or base["body"].strip()
                    payload = {"subject": subj, "body": bod}
                    thought = "I will run the ML pipeline to get predictions."
                else:
                    tool = "create_clickup_task"
                    base = json.loads(input_json)
                    subj_clean = (last_clean or {}).get("subject")
                    body_clean = (last_clean or {}).get("body")
                    subj = (subj_clean if (subj_clean or "").strip() else base["subject"]) or ""
                    bod = (body_clean if (body_clean or "").strip() else base["body"]) or ""
                    subj = subj.strip() or base["subject"].strip()
                    bod = bod.strip() or base["body"].strip()
                    pred = last_pred or {}
                    payload = {"subject": subj, "body": bod, "pred": pred}
                    thought = "I will create the ClickUp task and set fields."

        # Call the chosen tool
        obs_str = _call_tool(tool, payload)
        obs = _obs_json(obs_str)

        # Append to scratchpad
        scratch += (
            f"Thought: {thought}\n"
            f"Action: {tool}\n"
            f"Action Input: {json.dumps(payload, ensure_ascii=False)}\n"
            f"Observation: {obs_str}\n"
        )

        # Stop-on-error: return the tool's JSON as-is
        if isinstance(obs, dict) and obs.get("ok") is False:
            return obs

        # Track progress
        if tool == "clean_text" and obs.get("ok"):
            last_clean = {"subject": obs.get("subject"), "body": obs.get("body")}
        if tool == "predict_pipeline" and obs.get("ok"):
            last_pred = obs.get("pred") if isinstance(obs.get("pred"), dict) else None

        # Success on final step:
        if tool == "create_clickup_task":
            # Primary: explicit ok==True
            if isinstance(obs, dict) and obs.get("ok") is True:
                return {"ok": True, "task_id": obs.get("task_id"), "task_url": obs.get("task_url")}
            # Secondary: tolerate missing "ok" if task_id/url present
            if isinstance(obs, dict) and (obs.get("task_id") or obs.get("task_url")):
                return {"ok": True, "task_id": obs.get("task_id"), "task_url": obs.get("task_url")}

    # If we exit without success, surface the last observation's reason if present
    last_json = _obs_json(scratch.split("Observation:")[-1]) if "Observation:" in scratch else {}
    if isinstance(last_json, dict) and last_json.get("reason"):
        return {"ok": False, "reason": last_json.get("reason")}

    return {"ok": False, "reason": "agent_incomplete"}
