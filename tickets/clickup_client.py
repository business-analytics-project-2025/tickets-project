from __future__ import annotations

import os
import time
import json
import typing as t
import requests

from .config import TYPE_FIELD_ID, DEPT_FIELD_ID

# Configuration / constants
CLICKUP_BASE = os.getenv("CLICKUP_BASE", "https://api.clickup.com/api/v2")
CLICKUP_TOKEN = os.getenv("CLICKUP_TOKEN", "")
CLICKUP_LIST_ID = os.getenv("CLICKUP_LIST_ID", "")
CLICKUP_TEAM_ID = os.getenv("CLICKUP_TEAM_ID", "")

TIMEOUT = 30
HEADERS = {
    "Authorization": CLICKUP_TOKEN or "",
    "Content-Type": "application/json",
}

# Simple in-memory cache for field options to reduce API calls
_FIELD_OPTIONS_CACHE: dict[tuple[str, str], list[dict]] = {}


# Errors
class ClickUpHTTPError(RuntimeError):
    """Raised for non-2xx ClickUp responses with context."""
    def __init__(self, status: int, where: str, text: str):
        super().__init__(f"[{where}] HTTP {status}: {text[:500]}")
        self.status = status
        self.where = where
        self.text = text


def _require_env():
    missing = []
    if not CLICKUP_TOKEN:
        missing.append("CLICKUP_TOKEN")
    if not CLICKUP_LIST_ID:
        missing.append("CLICKUP_LIST_ID")
    if missing:
        raise ClickUpHTTPError(0, "env_check", f"Missing env vars: {', '.join(missing)}")


# HTTP helpers
def _req(method: str, path: str, where: str, *, params: dict | None = None, data: dict | None = None) -> dict:
    url = f"{CLICKUP_BASE}{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=HEADERS,
            params=params or None,
            data=json.dumps(data) if data is not None else None,
            timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise ClickUpHTTPError(-1, where, f"network_error: {e}") from e

    if resp.status_code // 100 != 2:
        raise ClickUpHTTPError(resp.status_code, where, resp.text or "<empty>")

    try:
        return resp.json() if resp.text else {}
    except Exception:
        return {}


# Core task helpers
def create_task(name: str, description: str, priority_num: int) -> dict:
    """
    Create a task in the configured list.
    Returns the task JSON (id, url, etc.).
    """
    _require_env()
    data = {
        "name": name or "(no subject)",
        "description": description or "",
        "priority": int(priority_num) if priority_num else 3,
    }
    return _req("POST", f"/list/{CLICKUP_LIST_ID}/task", "create_task", data=data)


def get_task(task_id: str) -> dict:
    return _req("GET", f"/task/{task_id}", "get_task")


def update_task_description(task_id: str, new_description: str) -> dict:
    data = {"description": new_description}
    return _req("PUT", f"/task/{task_id}", "update_task_description", data=data)


def _append_to_description(task_id: str, footer: str) -> None:
    """Append a small note to the end of the task description (best-effort)."""
    try:
        tdata = get_task(task_id)
        desc = tdata.get("description") or ""
        sep = "\n\n" if desc and not desc.endswith("\n") else ""
        update_task_description(task_id, desc + sep + footer)
    except ClickUpHTTPError:
        pass


def append_field_note(task_id: str, field_name: str, requested: str, chosen_display: str) -> None:
    footer = f"Note: {field_name} → requested '{requested}' → set '{chosen_display}'."
    _append_to_description(task_id, footer)


def append_tags_note(task_id: str, failed_tags: list[str]) -> None:
    foot = f"Note: Tags → requested {failed_tags} (some may not have been applied)."
    _append_to_description(task_id, foot)


# Tags
def _ensure_tag_exists(tag: str) -> None:
    """
    Try to create a tag at team scope if it doesn't exist.
    ClickUp allows POST /team/{team_id}/tag with {"tag": "<name>"}.
    Failures are ignored (best effort).
    """
    if not tag or not CLICKUP_TEAM_ID:
        return
    try:
        _req("POST", f"/team/{CLICKUP_TEAM_ID}/tag", "ensure_tag_exists", data={"tag": tag})
    except ClickUpHTTPError:
        pass


def add_tags(task_id: str, tags: list[str]) -> list[str]:
    """
    Ensure tags exist and attach them to the task.
    Returns list of tags that failed to attach (best effort).
    """
    failed: list[str] = []
    if not tags:
        return failed

    for tg in tags:
        if not tg:
            continue
        name = str(tg).strip()
        if not name:
            continue
        _ensure_tag_exists(name)
        try:
            _req("POST", f"/task/{task_id}/tag/{requests.utils.quote(name)}", "add_tag_to_task")
        except ClickUpHTTPError:
            failed.append(name)

    return failed


# Custom fields (dropdown)
def get_field_options_for_list(list_id: str, field_id: str) -> list[dict]:
    """
    Return list of options for a given custom dropdown field in a list.
    Caches results in memory.
    Each option is like: {"id": "...uuid...", "name": "Tech Support"}
    """
    cache_key = (list_id, field_id)
    if cache_key in _FIELD_OPTIONS_CACHE:
        return _FIELD_OPTIONS_CACHE[cache_key]

    response_data = _req("GET", f"/list/{list_id}/field", "get_list_fields")
    fields = response_data.get("fields", [])
    
    options: list[dict] = []
    for f in fields or []:
        if str(f.get("id")) == str(field_id):
            cfg = f.get("type_config") or {}
            for opt in (cfg.get("options") or []):
                label = opt.get("label") or opt.get("name") or ""
                options.append({"id": opt.get("id"), "name": label})
            break
    _FIELD_OPTIONS_CACHE[cache_key] = options
    return options


def _resolve_dropdown_option(list_id: str, field_id: str, requested_label: str) -> tuple[str | None, bool, str]:
    """
    Resolve a human label to option id.
    Returns (option_id, exact, chosen_display).
    - exact True when case-insensitive exact label match
    - if not exact, we try a light fuzzy (startswith/contains) and mark exact=False
    """
    label = (requested_label or "").strip()
    if not label:
        return None, False, ""

    opts = get_field_options_for_list(list_id, field_id)
    for o in opts:
        if (o.get("name") or "").strip().casefold() == label.casefold():
            return o.get("id"), True, o.get("name") or label

    for o in opts:
        nm = (o.get("name") or "")
        if nm.lower().startswith(label.lower()) or label.lower().startswith(nm.lower()):
            return o.get("id"), False, nm
    for o in opts:
        nm = (o.get("name") or "")
        if label.lower() in nm.lower():
            return o.get("id"), False, nm

    return None, False, label


def set_dropdown_value(task_id: str, field_id: str, requested_label: str) -> tuple[bool, bool, str, dict]:
    """
    Set a dropdown custom field by label.
    Returns (ok, exact, chosen_display, meta).
    - ok: True if ClickUp accepted the call (including no-op because already set)
    - exact: case-insensitive exact label match
    - chosen_display: the label we ended up writing (or pre-existing)
    - meta: extra info (option_id, request)
    """
    _require_env()

    option_id, exact, chosen = _resolve_dropdown_option(CLICKUP_LIST_ID, field_id, requested_label)
    meta: dict[str, t.Any] = {"requested": requested_label, "option_id": option_id, "exact": exact}

    if option_id is None:
        return True, False, chosen or requested_label, meta

    data = {"value": option_id}
    _req("PUT", f"/task/{task_id}/field/{field_id}", "set_dropdown_value", data=data)

    return True, exact, chosen, meta

def verify_custom_fields():
    """
    Checks if the configured Custom Field IDs exist on the configured List.
    Raises SystemExit on failure.
    """
    print("--- Verifying ClickUp Configuration ---")
    if not CLICKUP_TOKEN or not CLICKUP_LIST_ID:
        print("❌ CONFIGURATION ERROR: CLICKUP_TOKEN and CLICKUP_LIST_ID must be set.")
        raise SystemExit(1)

    try:
        response_data = _req("GET", f"/list/{CLICKUP_LIST_ID}/field", "verify_fields")
        fields_on_list = response_data.get("fields", [])
        field_ids_on_list = {field.get("id") for field in fields_on_list}

        type_id_ok = TYPE_FIELD_ID in field_ids_on_list
        dept_id_ok = DEPT_FIELD_ID in field_ids_on_list

        if type_id_ok:
            print(f"✅ 'Type' field ID ({TYPE_FIELD_ID}) found on list.")
        else:
            print(f"❌ CONFIGURATION ERROR: 'Type' field ID ({TYPE_FIELD_ID}) was NOT found on List {CLICKUP_LIST_ID}.")
        
        if dept_id_ok:
            print(f"✅ 'Department' field ID ({DEPT_FIELD_ID}) found on list.")
        else:
            print(f"❌ CONFIGURATION ERROR: 'Department' field ID ({DEPT_FIELD_ID}) was NOT found on List {CLICKUP_LIST_ID}.")

        if not type_id_ok or not dept_id_ok:
            print("\nPlease run find_field_ids.py again and update tickets/config.py with the correct IDs.")
            raise SystemExit(1)
        
        print("✅ ClickUp configuration is correct.")

    except Exception as e:
        print(f"❌ An API error occurred during verification: {e}")
        raise SystemExit(1)
