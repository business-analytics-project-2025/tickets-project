# agent_runner.py
from __future__ import annotations
import json, re
from typing import Dict, Any, List, Tuple

# LangChain (ReAct with ChatOllama / local Ollama)
from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Our helper + your existing modules
from text_clean import clean_subject_body

# Pipeline (use your async Orchestrator wrapper)
from tickets.orchestrator import predict_all as pipeline_predict_all

# ClickUp helpers + IDs & mapping reused from your code
from tickets.clickup_client import (
    create_task, add_tags, set_dropdown_value,
    append_tags_note, append_field_note, ClickUpHTTPError
)
from tickets.config import PRIORITY_TO_CLICKUP
from tickets.service_submit import TYPE_FIELD_ID, DEPT_FIELD_ID, submit_ticket as service_submit

from tickets.duplicate_check import dedupe as dc_dedupe, remember as dc_remember, make_hash as dc_make_hash

from tickets.clickup_client import get_task

# -------------------- Tool functions (string in / out as JSON) --------------------

def _as_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def tool_clean_text(inp: str) -> str:
    """
    Expects: {"subject": "...", "body": "..."}
    Returns: {"ok": true, "subject": "...", "body": "..."} OR {"ok": false, "reason": "..."}
    """
    try:
        data = json.loads(inp or "{}")
        s, b = clean_subject_body(data.get("subject", ""), data.get("body", ""))
        return _as_json({"ok": True, "subject": s, "body": b})
    except Exception as e:
        return _as_json({"ok": False, "reason": f"clean_failed: {e}"})

def tool_predict_pipeline(inp: str) -> str:
    # Expects: {"subject":"...","body":"..."}
    import json
    try:
        data = json.loads(inp or "{}")
        subject = data.get("subject", "")
        body = data.get("body", "")
        pred = pipeline_predict_all(subject, body)
        return json.dumps({"ok": True, "pred": pred}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"prediction_failed: {e}"}, ensure_ascii=False)

def tool_create_clickup_task(inp: str) -> str:
    """
    Expects:
      {"subject":"...", "body":"...", "pred":{
          "priority": str, "type": str, "department": str, "tags": [str, ...]
      }}
    Returns:
      {"ok": true, "task_id": "<id>", "task_url": "<url>"}  (success)
      {"ok": false, "reason": "<why>"}                      (stop-on-error)
    """
    import json
    try:
        data = json.loads(inp or "{}")
        subject = (data.get("subject") or "").strip() or "(no subject)"
        body    = (data.get("body") or "").strip()
        pred    = data.get("pred") or {}

        # Priority mapping (default Normal=3)
        priority_str = (pred.get("priority") or "").strip()
        priority_num = PRIORITY_TO_CLICKUP.get(priority_str, 3)

        # --- Create task (short id is fine with your client) ---
        task = create_task(name=subject, description=body, priority_num=priority_num)
        task_id  = str(task.get("id") or "").strip()
        task_url = str(task.get("url") or "").strip()
        if not task_id:
            return _as_json({"ok": False, "reason": "clickup_failed: create_task returned no id"})

        # --- TYPE dropdown ---
        tval = (pred.get("type") or "").strip()
        if tval:
            try:
                _soft_set_dropdown(task_id, TYPE_FIELD_ID, "Type", tval)
            except ClickUpHTTPError as e:
                return _as_json({"ok": False, "reason": f"clickup_failed: type_set: {e}"})

        # --- DEPARTMENT dropdown ---
        dval = (pred.get("department") or "").strip()
        if dval:
            try:
                _soft_set_dropdown(task_id, DEPT_FIELD_ID, "Department", dval)
            except ClickUpHTTPError as e:
                return _as_json({"ok": False, "reason": f"clickup_failed: department_set: {e}"})


        # --- TAGS (best effort; falls back to note on failure) ---
        tags = pred.get("tags") or []
        try:
            failed = add_tags(task_id, tags)
            if failed:
                append_tags_note(task_id, failed)
        except Exception:
            append_tags_note(task_id, tags or [])

        # --- Remember duplicate hash AFTER success ---
        try:
            h = dc_make_hash(subject, body)
            dc_remember(h)
        except Exception:
            pass

        return _as_json({"ok": True, "task_id": task_id, "task_url": task_url})

    except Exception as e:
        return _as_json({"ok": False, "reason": f"clickup_failed: create_or_set: {e}"})



def tool_check_duplicate(inp: str) -> str:
    """
    Input JSON: {"subject": "...", "body": "..."}  (use CLEANED text)
    Returns:
      - Duplicate: {"ok": false, "reason": "duplicate_found", "dup_hash": "<sha256>"}
        (Agent will STOP due to ok:false per the hard rule.)
      - No duplicate: {"ok": true, "duplicate": false, "dup_hash": "<sha256>"}
      - Error: {"ok": false, "reason": "duplicate_check_failed: ..."}
    """
    import json
    try:
        data = json.loads(inp or "{}")
        subject = (data.get("subject") or "").strip()
        body    = (data.get("body") or "").strip()
        is_dup, h = dc_dedupe(subject, body)  # your function: returns (bool, hash)
        if is_dup:
            return json.dumps({"ok": False, "reason": "duplicate_found", "dup_hash": h}, ensure_ascii=False)
        return json.dumps({"ok": True, "duplicate": False, "dup_hash": h}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"duplicate_check_failed: {e}"}, ensure_ascii=False)
    
# --- helper: tolerant dropdown setter (treats read-back lag as success) ---
def _soft_set_dropdown(task_id: str, field_id: str, field_label_for_note: str, desired_value: str) -> None:
    """
    Calls tickets.clickup_client.set_dropdown_value(). If ClickUp returns 2xx but
    read-back is still None (eventual consistency), we add a small '(pending)'
    note and treat it as success so the agent doesn't fail the run.
    """
    from tickets.clickup_client import set_dropdown_value, append_field_note, ClickUpHTTPError

    try:
        _resp, exact, chosen, _optid = set_dropdown_value(task_id, field_id, desired_value)
        if not exact and chosen:
            append_field_note(task_id, f"{field_label_for_note} (fuzzy)", desired_value, chosen)
    except ClickUpHTTPError as e:
        msg = str(e)
        # the specific soft case from your client
        if "Dropdown write returned 2xx but value not persisted" in msg:
            append_field_note(task_id, f"{field_label_for_note} (pending/verify)", desired_value, "(pending)")
            return  # treat as success
        # anything else is a real failure -> bubble up
        raise

def _cf_value(task_json: dict, field_id: str):
    for cf in (task_json or {}).get("custom_fields", []) or []:
        if str(cf.get("id")) == str(field_id):
            return cf.get("value")
    return None

def _blocking_wait_dropdown(task_id: str, field_id: str, expect_label: str, expect_opt_id: str | None) -> bool:
    """
    Wait up to ~12s for a dropdown to be visible on the task.
    Accepts equality by option id OR by label (string or object).
    Returns True if visible, False if not.
    """
    import time
    deadline = time.time() + 12.0
    while time.time() < deadline:
        try:
            t = get_task(task_id)
        except ClickUpHTTPError:
            time.sleep(0.7)
            continue
        val = _cf_value(t, field_id)
        # accept option id exact
        if expect_opt_id and val is not None and str(val) == str(expect_opt_id):
            return True
        # accept label as string
        if isinstance(val, str) and val.strip().lower() == (expect_label or "").strip().lower():
            return True
        # accept object with id/label/value
        if isinstance(val, dict):
            pid = val.get("id")
            plabel = (val.get("label") or val.get("value") or "").strip()
            if (expect_opt_id and str(pid) == str(expect_opt_id)) or (plabel and plabel.lower() == (expect_label or "").lower()):
                return True
        time.sleep(1.2)
    return False
# -------------------- Agent setup --------------------

ALLOWED_TOOLS = ["clean_text", "check_duplicate", "predict_pipeline", "create_clickup_task"]

SYSTEM_PROMPT = (
    "You are a disciplined ReAct agent that MUST use tools.\n"
    "Protocol (exactly):\n"
    "Thought: describe next step\n"
    "Action: one of [clean_text, check_duplicate, predict_pipeline, create_clickup_task]\n"
    "Action Input: a single JSON object (no code fences)\n"
    "Observation: the tool's JSON result\n"
    "…repeat until done… then:\n"
    "Final Answer: a single JSON object only.\n\n"
    "Hard rules:\n"
    "1) Always call clean_text FIRST.\n"
    "2) Then call check_duplicate. If it returns {\"ok\": false, \"reason\": \"duplicate_found\", ...}, STOP and output that JSON as the Final Answer.\n"
    "3) If no duplicate, call predict_pipeline, then create_clickup_task, and STOP.\n"
    "4) The Final Answer must be ONLY one JSON object (no extra text, no code fences).\n"
    "5) Never invent tool names; use only [clean_text, check_duplicate, predict_pipeline, create_clickup_task].\n"
    "6) Always provide valid JSON in Action Input (no comments, no trailing commas).\n\n"
    "Mini example (duplicate → stop):\n"
    "Thought: I will clean the text.\n"
    "Action: clean_text\n"
    "Action Input: {\"subject\":\"s\",\"body\":\"b\"}\n"
    "Observation: {\"ok\": true, \"subject\": \"s2\", \"body\": \"b2\"}\n"
    "Thought: I will check for duplicates.\n"
    "Action: check_duplicate\n"
    "Action Input: {\"subject\":\"s2\",\"body\":\"b2\"}\n"
    "Observation: {\"ok\": false, \"reason\": \"duplicate_found\", \"dup_hash\": \"...\"}\n"
    "Final Answer: {\"ok\": false, \"reason\": \"duplicate_found\", \"dup_hash\": \"...\"}\n\n"
    "Mini example (success):\n"
    "Thought: I will clean the text.\n"
    "Action: clean_text\n"
    "Action Input: {\"subject\":\"s\",\"body\":\"b\"}\n"
    "Observation: {\"ok\": true, \"subject\": \"s2\", \"body\": \"b2\"}\n"
    "Thought: No duplicate; I will run the ML pipeline.\n"
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
    return (
        SYSTEM_PROMPT
        + "\n"
        + "Input ticket JSON:\n"
        + input_json
        + "\n\n"
        + (scratchpad or "")
    )

_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.+?)\s*"
    r"Action:\s*(?P<tool>\w+)\s*"
    r"Action Input:\s*(?P<input>\{[\s\S]*\})",
    re.IGNORECASE | re.DOTALL
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

def _obs_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # last resort: pull first {...}
        m = re.search(r"\{[\s\S]*\}", s or "")
        return json.loads(m.group(0)) if m else {"ok": False, "reason": "non_json_observation"}

def _call_tool(tool: str, payload: Dict[str, Any]) -> str:
    if tool == "clean_text":
        return tool_clean_text(json.dumps(payload, ensure_ascii=False))
    if tool == "predict_pipeline":
        return tool_predict_pipeline(json.dumps(payload, ensure_ascii=False))
    if tool == "create_clickup_task":
        return tool_create_clickup_task(json.dumps(payload, ensure_ascii=False))
    return json.dumps({"ok": False, "reason": f"invalid_tool: {tool}"}, ensure_ascii=False)

def run_agent(subject: str, body: str) -> Dict[str, Any]:
    llm = ChatOllama(model="llama3", temperature=0.0, num_ctx=4096, num_predict=512)
    input_json = json.dumps({"subject": subject, "body": body}, ensure_ascii=False)

    scratch = ""  # ReAct transcript accumulated here
    last_clean = None
    last_pred = None

    # Deterministic 4-step loop
    for step in range(1, 5):
        # Ask the LLM for the next Thought/Action
        resp = llm.invoke(_render_prompt(input_json, scratch))
        text = getattr(resp, "content", "") if resp else ""

        # Try to parse the next Action
        try:
            tool, payload, thought = _parse_action(text)
        except Exception as e:
            # One focused retry with explicit template
            nudge = (
                f"{scratch}"
                "Your previous message did not include a valid Action / Action Input JSON.\n"
                "Respond in EXACTLY this format (no code fences):\n"
                "Thought: <your short thought>\n"
                f"Action: <one of {ALLOWED_TOOLS}>\n"
                "Action Input: {\"key\":\"value\",...}\n"
            )
            resp2 = llm.invoke(_render_prompt(input_json, nudge))
            text2 = getattr(resp2, "content", "") if resp2 else ""
            try:
                tool, payload, thought = _parse_action(text2)
            except Exception:
                # Enforce step tool if parsing still fails
                if step == 1 and tool != "clean_text":
                    tool = "clean_text"
                    payload = json.loads(input_json)

                elif step == 2 and tool != "check_duplicate":
                    tool = "check_duplicate"
                    subj = (last_clean or {}).get("subject") or json.loads(input_json)["subject"]
                    bod  = (last_clean or {}).get("body") or json.loads(input_json)["body"]
                    payload = {"subject": subj, "body": bod}

                elif step == 3 and tool != "predict_pipeline":
                    tool = "predict_pipeline"
                    subj = (last_clean or {}).get("subject") or json.loads(input_json)["subject"]
                    bod  = (last_clean or {}).get("body") or json.loads(input_json)["body"]
                    payload = {"subject": subj, "body": bod}

                elif step == 4 and tool != "create_clickup_task":
                    tool = "create_clickup_task"
                    subj = (last_clean or {}).get("subject") or json.loads(input_json)["subject"]
                    bod  = (last_clean or {}).get("body") or json.loads(input_json)["body"]
                    pred = last_pred or {}
                    payload = {"subject": subj, "body": bod, "pred": pred}

        # Enforce order: clean_text → predict_pipeline → create_clickup_task
        if step == 1 and tool != "clean_text":
            tool = "clean_text"
            payload = json.loads(input_json)
        if step == 2 and tool != "predict_pipeline":
            tool = "predict_pipeline"
            # prefer cleaned values if we have them
            subj = (last_clean or {}).get("subject") or json.loads(input_json)["subject"]
            bod  = (last_clean or {}).get("body") or json.loads(input_json)["body"]
            payload = {"subject": subj, "body": bod}
        if step == 3 and tool != "create_clickup_task":
            tool = "create_clickup_task"
            subj = (last_clean or {}).get("subject") or json.loads(input_json)["subject"]
            bod  = (last_clean or {}).get("body") or json.loads(input_json)["body"]
            pred = last_pred or {}
            payload = {"subject": subj, "body": bod, "pred": pred}

        # Call the chosen tool
        obs_str = _call_tool(tool, payload)
        obs = _obs_json(obs_str)

        # Append to scratchpad
        scratch += (
            f"Thought: {('I will ' + ('clean the text' if tool=='clean_text' else 'run the ML pipeline' if tool=='predict_pipeline' else 'create the ClickUp task'))}.\n"
            f"Action: {tool}\n"
            f"Action Input: {json.dumps(payload, ensure_ascii=False)}\n"
            f"Observation: {obs_str}\n"
        )

        # Stop-on-error
        if isinstance(obs, dict) and obs.get("ok") is False:
            return obs

        # Track progress
        if tool == "clean_text" and obs.get("ok"):
            last_clean = {"subject": obs.get("subject"), "body": obs.get("body")}
        if tool == "predict_pipeline" and obs.get("ok"):
            last_pred = obs.get("pred") if isinstance(obs.get("pred"), dict) else None

        # If task created, finish
        if tool == "create_clickup_task" and obs.get("ok"):
            return {"ok": True, "task_id": obs.get("task_id"), "task_url": obs.get("task_url")}

    # If we exit loop without success, try to salvage a JSON the model might have printed
    # (stays agentic: we only parse model/observation text; no direct tool calls)
    m = re.search(r"\{[\s\S]*\}", scratch.split("Observation:")[-1] if "Observation:" in scratch else "")
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"ok": False, "reason": "agent_incomplete"}


