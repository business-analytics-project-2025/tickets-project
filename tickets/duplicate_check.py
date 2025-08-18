# tickets/duplicate_check.py
import hashlib, json, time
from pathlib import Path
from typing import Tuple

CACHE_PATH = Path(".dup_cache.json")
TTL_SECONDS = 7 * 24 * 3600  # one week

def _now() -> int:
    return int(time.time())

def _load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")

def make_hash(subject: str, body: str) -> str:
    key = (subject or "").strip() + "||" + (body or "").strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def is_duplicate(hash_str: str) -> bool:
    cache = _load_cache()
    entry = cache.get(hash_str)
    if not entry:
        return False
    # TTL expiry
    if _now() - entry.get("ts", 0) > TTL_SECONDS:
        del cache[hash_str]
        _save_cache(cache)
        return False
    return True

def remember(hash_str: str):
    cache = _load_cache()
    cache[hash_str] = {"ts": _now()}
    _save_cache(cache)

def dedupe(subject: str, body: str) -> Tuple[bool, str]:
    """Returns (is_dup, hash_str)."""
    h = make_hash(subject, body)
    return is_duplicate(h), h
