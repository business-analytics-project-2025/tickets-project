# tickets/clickup_client.py
import os, hashlib, requests
from typing import Dict, List, Optional

BASE = "https://api.clickup.com/api/v2"
CLICKUP_TOKEN = os.getenv("CLICKUP_TOKEN")
CLICKUP_LIST_ID = os.getenv("CLICKUP_LIST_ID", "901514139552")  # your list
CLICKUP_TEAM_ID = os.getenv("CLICKUP_TEAM_ID", "90151452884")   # your team

HEADERS = {
    "Authorization": CLICKUP_TOKEN or "",
    "Content-Type": "application/json",
}

class ClickUpError(RuntimeError):
    pass

def _check_token():
    if not CLICKUP_TOKEN:
        raise ClickUpError("CLICKUP_TOKEN is not set in env.")

def _raise_for_status(r: requests.Response):
    try:
        r.raise_for_status()
    except Exception as e:
        raise ClickUpError(f"{r.status_code} {r.text}") from e

def ensure_workspace_tag(name: str):
    """Create a workspace tag if missing (idempotent)."""
    _check_token()
    # ClickUp 'create tag' payload shape:
    r = requests.post(
        f"{BASE}/team/{CLICKUP_TEAM_ID}/tag",
        headers=HEADERS,
        json={"tag": {"name": name}},
        timeout=15,
    )
    if r.status_code not in (200, 409):  # 409 means already exists
        _raise_for_status(r)

def add_tags(task_id: str, tags: List[str]):
    """Attach tags to a task, auto-creating unknown ones."""
    _check_token()
    for t in tags:
        r = requests.post(f"{BASE}/task/{task_id}/tag/{t}", headers=HEADERS, timeout=15)
        if r.status_code == 404:
            ensure_workspace_tag(t)
            r = requests.post(f"{BASE}/task/{task_id}/tag/{t}", headers=HEADERS, timeout=15)
        _raise_for_status(r)

def create_task(name: str, description: str, priority_num: int) -> Dict:
    """Create a task in your list; returns {'id':..., 'url':...}."""
    _check_token()
    r = requests.post(
        f"{BASE}/list/{CLICKUP_LIST_ID}/task",
        headers=HEADERS,
        json={"name": name, "description": description, "priority": priority_num},
        timeout=20,
    )
    _raise_for_status(r)
    data = r.json()
    return {"id": data["id"], "url": data.get("url", "")}

def list_fields(list_id: Optional[str] = None) -> Dict:
    _check_token()
    lid = list_id or CLICKUP_LIST_ID
    r = requests.get(f"{BASE}/list/{lid}/field", headers=HEADERS, timeout=15)
    _raise_for_status(r)
    return r.json()

def _get_dropdown_options(field_id: str) -> Dict[str, str]:
    """Return name->id map of existing dropdown options."""
    fields = list_fields()
    opts = {}
    for f in fields.get("fields", []):
        if f.get("id") == field_id and f.get("type") == "drop_down":
            for o in f.get("type_config", {}).get("options", []):
                opts[o["name"]] = o["id"]
    return opts

def ensure_dropdown_option(field_id: str, option_name: str) -> str:
    """Ensure a dropdown option exists; create and return its option id."""
    _check_token()
    options = _get_dropdown_options(field_id)
    if option_name in options:
        return options[option_name]
    # Add option
    r = requests.post(
        f"{BASE}/field/{field_id}/option",
        headers=HEADERS,
        json={"name": option_name},
        timeout=15,
    )
    _raise_for_status(r)
    new_opt = r.json()
    return new_opt["id"]

def set_dropdown_value(task_id: str, field_id: str, option_name: str):
    """Set a dropdown custom field; auto-add option if missing."""
    opt_id = ensure_dropdown_option(field_id, option_name)
    r = requests.post(
        f"{BASE}/task/{task_id}/field/{field_id}",
        headers=HEADERS,
        json={"value": opt_id},
        timeout=15,
    )
    _raise_for_status(r)

def task_exists_with_hash(hash_str: str) -> bool:
    """Optional: quick client-side cache; real ClickUp search can be added later."""
    # For first iteration, we rely on a local cache approach (see duplicate_check.py).
    # If you want API-based search, we can implement a 'GET /team/{team_id}/task' filter later.
    return False
