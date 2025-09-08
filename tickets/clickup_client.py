# tickets/clickup_client.py
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

import requests

BASE = "https://api.clickup.com/api/v2"

# ---------- HTTP helper & errors ----------

def _get_headers() -> Dict[str, str]:
    token = os.environ.get("CLICKUP_TOKEN")
    if not token:
        raise RuntimeError("CLICKUP_TOKEN is not set in environment")
    return {"Authorization": token, "Content-Type": "application/json"}

class ClickUpHTTPError(RuntimeError):
    def __init__(
        self,
        msg: str,
        *,
        status: Optional[int] = None,
        url: Optional[str] = None,
        payload: Optional[dict] = None,
        resp_text: Optional[str] = None,
    ):
        super().__init__(msg)
        self.status = status
        self.url = url
        self.payload = payload
        self.resp_text = resp_text

    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.url:
            parts.append(f"url={self.url}")
        if self.payload is not None:
            parts.append(f"payload={self.payload}")
        if self.resp_text:
            trimmed = self.resp_text if len(self.resp_text) < 2000 else self.resp_text[:2000] + "…"
            parts.append(f"resp={trimmed}")
        return base + (" [" + " | ".join(parts) + "]" if parts else "")

def _req(method: str, url: str, json: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    r = requests.request(method, url, headers=_get_headers(), json=json, params=params, timeout=30)
    if r.status_code // 100 != 2:
        raise ClickUpHTTPError(
            f"HTTP {r.status_code} {method} {url}",
            status=r.status_code, url=url, payload=json, resp_text=r.text
        )
    try:
        return r.json() if r.text else {}
    except Exception:
        return {}

def _params_task() -> dict:
    """
    If your workspace uses short/custom task ids (like '86c5c...'),
    ClickUp requires these query params for ALL /task/{id} calls.
    Safe to include even when using numeric ids.
    """
    team_id = os.environ.get("CLICKUP_TEAM_ID")
    return {"custom_task_ids": "true", "team_id": team_id} if team_id else {}

# ---------- Env helpers ----------

def _get_list_id() -> str:
    v = os.environ.get("CLICKUP_LIST_ID")
    if not v:
        raise RuntimeError("CLICKUP_LIST_ID is not set in environment")
    return v

def _get_team_id() -> str:
    v = os.environ.get("CLICKUP_TEAM_ID")
    if not v:
        raise RuntimeError("CLICKUP_TEAM_ID is not set in environment")
    return v

# ---------- IDs discovery (Space for this List) ----------

_SPACE_ID_CACHE: Optional[str] = None

def _get_space_id_for_list() -> str:
    """
    Resolve and cache Space ID for the configured List.
    GET /list/{list_id} → {'space': {'id': ...}} (or nested under 'list')
    """
    global _SPACE_ID_CACHE
    if _SPACE_ID_CACHE:
        return _SPACE_ID_CACHE
    list_id = _get_list_id()
    url = f"{BASE}/list/{list_id}"
    data = _req("GET", url)
    space_id = (
        (data.get("space") or {}).get("id")
        or ((data.get("list") or {}).get("space") or {}).get("id")
    )
    if not space_id:
        raise ClickUpHTTPError("Could not resolve space.id from List", url=url, resp_text=str(data))
    _SPACE_ID_CACHE = str(space_id)
    return _SPACE_ID_CACHE

# ---------- Public helpers ----------

def list_fields() -> dict:
    """Return field metadata for the current List (debug/ops helper)."""
    list_id = _get_list_id()
    url = f"{BASE}/list/{list_id}/field"
    return _req("GET", url)

def create_task(name: str, description: str, priority_num: int) -> dict:
    """
    Create a task on the configured List.
    priority: 1 urgent, 2 high, 3 normal, 4 low.
    Returns the task JSON (expects 'id' (short/custom ok) and 'url').
    """
    list_id = _get_list_id()
    url = f"{BASE}/list/{list_id}/task"
    payload = {"name": name, "description": description, "priority": priority_num}
    return _req("POST", url, json=payload)

def get_task(task_id: str) -> dict:
    url = f"{BASE}/task/{task_id}"
    return _req("GET", url, params=_params_task())

def update_task_description(task_id: str, new_description: str) -> dict:
    url = f"{BASE}/task/{task_id}"
    return _req("PUT", url, json={"description": new_description}, params=_params_task())

# ---------- Dropdowns: matching + robust write + tolerant read-back verification ----------

_FIELD_CACHE: dict[str, dict] = {}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _get_list_fields() -> dict:
    list_id = _get_list_id()
    url = f"{BASE}/list/{list_id}/field"
    return _req("GET", url)

def _find_dropdown(field_id: str) -> dict:
    global _FIELD_CACHE
    if field_id in _FIELD_CACHE:
        return _FIELD_CACHE[field_id]

    data = _get_list_fields()
    field = next((f for f in data.get("fields", []) if f.get("id") == field_id), None)
    if not field:
        raise ClickUpHTTPError("Field not attached to list",
                               url=f"{BASE}/list/{_get_list_id()}/field",
                               resp_text=str(data))
    if field.get("type") != "drop_down":
        raise ClickUpHTTPError("Field is not drop_down", resp_text=str(field))

    _FIELD_CACHE[field_id] = field
    return field

def _dropdown_options(field_id: str) -> List[dict]:
    field = _find_dropdown(field_id)
    return field.get("type_config", {}).get("options", []) or []

def _option_index_for(field_id: str, option_id: str) -> Optional[int]:
    opts = _dropdown_options(field_id)
    for i, opt in enumerate(opts):
        if str(opt.get("id")) == str(option_id):
            if "orderindex" in opt and isinstance(opt["orderindex"], int):
                return int(opt["orderindex"])
            return i
    return None

def _option_label_for(field_id: str, option_id: str) -> Optional[str]:
    opts = _dropdown_options(field_id)
    for opt in opts:
        if str(opt.get("id")) == str(option_id):
            return (opt.get("name") or "").strip()
    return None

def _match_dropdown_option(field_id: str, value_name: str, similarity_threshold: float = 0.78) -> Tuple[str, str, bool]:
    """
    Map a predicted string to an existing dropdown option.
    Returns: (option_id, chosen_display_name, is_exact)
    If nothing reasonable is found, returns empty option_id (caller decides how to handle).
    """
    field = _find_dropdown(field_id)
    options = field.get("type_config", {}).get("options", []) or []
    names = [(opt.get("name") or "").strip() for opt in options]
    ids   = [opt.get("id") for opt in options]

    wanted_raw  = (value_name or "").strip()
    wanted_norm = _norm(wanted_raw)

    # 1) case-insensitive exact
    for i, n in enumerate(names):
        if n.lower() == wanted_raw.lower() and ids[i]:
            return ids[i], names[i], True

    # 2) normalized exact
    norm_to_idx = {_norm(n): i for i, n in enumerate(names)}
    if wanted_norm in norm_to_idx:
        i = norm_to_idx[wanted_norm]
        if ids[i]:
            return ids[i], names[i], True

    # 3) fuzzy
    best_i, best_sim = -1, 0.0
    for i, n in enumerate(names):
        sim = _similar(_norm(n), wanted_norm)
        if sim > best_sim:
            best_sim, best_i = sim, i
    if best_i >= 0 and best_sim >= similarity_threshold and ids[best_i]:
        return ids[best_i], names[best_i], False

    # 4) explicit failure
    return "", wanted_raw, False

def _set_dropdown_value_payloads(task_id: str, field_id: str, option_id: str, option_label: str) -> dict:
    """
    Try every known write strategy with every known value shape:
      endpoints:
        - POST /task/{id}/field/{field_id}
        - PUT  /task/{id}/field/{field_id}
        - PUT  /task/{id}   (custom_fields array)
      value shapes:
        - id          (string)
        - {"id": id}  (object)
        - index       (integer)
        - label       (string display name)
        - {"label": label}
    """
    field_url = f"{BASE}/task/{task_id}/field/{field_id}"
    task_url  = f"{BASE}/task/{task_id}"
    params = _params_task()

    idx = _option_index_for(field_id, option_id)

    value_shapes: List[dict] = []
    value_shapes.append({"value": option_id})
    value_shapes.append({"value": {"id": option_id}})
    if idx is not None:
        value_shapes.append({"value": idx})
    if option_label:
        value_shapes.append({"value": option_label})
        value_shapes.append({"value": {"label": option_label}})

    attempts: List[Tuple[str, str, dict]] = []
    for payload in value_shapes:
        attempts.append(("POST", field_url, payload))
        attempts.append(("PUT",  field_url, payload))
        # full-task update form
        attempts.append(("PUT",  task_url, {"custom_fields": [{"id": field_id, **payload}]}))

    last_err: Optional[ClickUpHTTPError] = None
    for method, url, payload in attempts:
        try:
            return _req(method, url, json=payload, params=params)
        except ClickUpHTTPError as e:
            last_err = e
            continue

    raise ClickUpHTTPError(
        "Failed to set dropdown after all endpoint/payload strategies",
        status=(last_err.status if last_err else None),
        url=(last_err.url if last_err else field_url),
        payload={"tried": [{"method": m, "url": u, "payload": p} for (m, u, p) in attempts]},
        resp_text=(last_err.resp_text if last_err else None),
    )

def _read_dropdown_value(task_id: str, field_id: str) -> Optional[Any]:
    """
    Return the current stored value for this dropdown on the task:
      - may be option id (string)
      - may be numeric index (int)
      - may be object with id or label
      - may be label string in some tenants
    """
    task = get_task(task_id)
    for cf in task.get("custom_fields", []) or []:
        if str(cf.get("id")) == str(field_id):
            val = cf.get("value")
            if isinstance(val, dict):
                return val.get("id") or val.get("label") or val.get("value")
            return val
    return None

def set_dropdown_value(
    task_id: str,
    field_id: str,
    value_name: str,
    *,
    allow_pending: bool = True,
    max_wait_s: float = 12.0,
) -> Tuple[dict, bool, str, str]:
    """
    Match predicted value to an existing option, write it using robust strategies,
    and verify by accepting id OR index OR label on read-back.
    Returns: (resp_json, is_exact, chosen_display_name, chosen_option_id)

    Soft-consistency handling:
      - If ClickUp returns 2xx but the value isn't readable yet, we retry up to ~max_wait_s.
      - If still not visible and allow_pending=True, we RETURN normally (do not raise).
      - If still not visible and allow_pending=False, we RAISE ClickUpHTTPError.
    """
    import time as _t

    opt_id, chosen_name, exact = _match_dropdown_option(field_id, value_name)
    if not opt_id:
        raise ClickUpHTTPError(
            f"No matching option for '{value_name}' on field {field_id}",
            payload={"predicted": value_name}
        )

    label = _option_label_for(field_id, opt_id) or chosen_name
    resp = _set_dropdown_value_payloads(task_id, field_id, opt_id, label)

    # --- read-back verification with backoff up to max_wait_s ---
    start = _t.time()

    def _persisted_ok() -> bool:
        val = _read_dropdown_value(task_id, field_id)
        # Accept exact id
        if val is not None and str(val) == str(opt_id):
            return True
        # Accept numeric index
        if isinstance(val, int):
            expected_idx = _option_index_for(field_id, opt_id)
            if expected_idx is not None and int(val) == int(expected_idx):
                return True
        # Accept label string
        if isinstance(val, str):
            if val.strip().lower() == (label or "").strip().lower():
                return True
        # Accept object with id/label/value
        if isinstance(val, dict):
            pid = val.get("id")
            plabel = (val.get("label") or val.get("value") or "").strip()
            if (pid and str(pid) == str(opt_id)) or (plabel and plabel.lower() == (label or "").lower()):
                return True
        return False

    # Backoff: 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, ... up to max_wait_s
    attempt = 0
    sleep = 0.5
    while (_t.time() - start) < max_wait_s:
        if _persisted_ok():
            return resp, exact, chosen_name, opt_id
        _t.sleep(sleep)
        attempt += 1
        sleep = min(sleep + 0.3, 2.0)

    # Still not visible after max_wait_s
    if allow_pending:
        # Return as success; callers may annotate "(pending/verify)"
        return resp, exact, chosen_name, opt_id

    # Strict mode: raise with diagnostic
    raise ClickUpHTTPError(
        "Dropdown write returned 2xx but value not persisted",
        url=f"{BASE}/task/{task_id}",
        payload={
            "field_id": field_id,
            "expected_option_id": opt_id,
            "expected_label": label,
            "expected_index": _option_index_for(field_id, opt_id),
            "read_back_value": _read_dropdown_value(task_id, field_id),
        },
    )

# ---------- Tags: ensure in Space, attach to task (best-effort) ----------

def _ensure_space_tag(tag_name: str) -> None:
    """
    Ensure a tag exists at the **Space** level for the List's Space.
    """
    space_id = _get_space_id_for_list()
    url = f"{BASE}/space/{space_id}/tag"

    last_err: Optional[ClickUpHTTPError] = None
    # Try both payload shapes seen across deployments
    for payload in ({"tag": {"name": tag_name}}, {"name": tag_name}):
        try:
            _req("POST", url, json=payload)  # many deployments 2xx on "already exists"
            return
        except ClickUpHTTPError as e:
            if e.status in (400, 409, 422):
                return
            last_err = e
            continue

    if last_err is not None:
        raise last_err

def add_tags(task_id: str, tags: List[str]) -> List[str]:
    """
    Ensure tags exist at Space scope, then attach each to the task.
    Returns: list of tags that could NOT be attached (for description fallback).
    """
    failed: List[str] = []
    for tag in tags or []:
        tag = (tag or "").strip()
        if not tag:
            continue

        # Ensure exists in Space
        try:
            _ensure_space_tag(tag)
        except ClickUpHTTPError:
            failed.append(tag)
            continue

        # Try path form: POST /task/{task_id}/tag/{tag}
        try:
            url1 = f"{BASE}/task/{task_id}/tag/{tag}"
            _req("POST", url1, params=_params_task())
            continue  # success
        except ClickUpHTTPError as e1:
            if e1.status not in (404, 405):
                failed.append(tag)
                continue

        # Body form: POST /task/{task_id}/tag  {"tags": ["..."]}
        try:
            url2 = f"{BASE}/task/{task_id}/tag"
            _req("POST", url2, json={"tags": [tag]}, params=_params_task())
        except ClickUpHTTPError as e2:
            if e2.status not in (400, 404, 405, 409, 422):
                failed.append(tag)
            else:
                failed.append(tag)

    return failed

# ---------- Description helpers (visibility fallbacks) ----------

def append_tags_note(task_id: str, tags: List[str]) -> None:
    """Append a 'Predicted tags (ML): …' note at the end of the description."""
    tags = [t for t in (tags or []) if t]
    if not tags:
        return
    task = get_task(task_id)
    desc = (task.get("description") or "").rstrip()
    note = "\n\n---\n**Predicted tags (ML):** " + ", ".join(sorted(set(tags)))
    new_desc = (desc + note) if desc else note.lstrip()
    update_task_description(task_id, new_desc)

def append_field_note(task_id: str, field_label: str, predicted_value: str, chosen_value: str) -> None:
    """
    Append a one-liner to the task description explaining the dropdown match or pending state.
    """
    predicted_value = (predicted_value or "").strip()
    if not predicted_value:
        return
    task = get_task(task_id)
    desc = (task.get("description") or "").rstrip()
    note = f'\n\n---\n**Predicted {field_label}:** "{predicted_value}" → set to **{chosen_value}**'
    new_desc = (desc + note) if desc else note.lstrip()
    update_task_description(task_id, new_desc)
