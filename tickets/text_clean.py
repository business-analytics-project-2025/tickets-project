import re
from typing import Tuple

_SIG_CUTOFF_PATTERNS = [
    r"^--\s*$",
    r"^thanks[,.! ]*$",
    r"^thank you[,.! ]*$",
    r"^best( regards)?[,.! ]*$",
    r"^regards[,.! ]*$",
    r"^sent from my iphone",
    r"^cheers[,.! ]*$",
]

_COMMON_TYPO_MAP = {
    "hte": "the",
    "teh": "the",
    "adress": "address",
    "recieve": "receive",
    "seperate": "separate",
    "occurence": "occurrence",
    "occured": "occurred",
    "enviroment": "environment",
    "definately": "definitely",
    "intermitently": "intermittently",
}

_WORD_RE = re.compile(r"\b(" + "|".join(map(re.escape, _COMMON_TYPO_MAP.keys())) + r")\b", re.IGNORECASE)

def _strip_signature(body: str) -> str:
    lines = body.splitlines()
    out = []
    cut = False
    for ln in lines:
        if any(re.match(p, ln.strip(), re.IGNORECASE) for p in _SIG_CUTOFF_PATTERNS):
            cut = True
            break
        out.append(ln)
    return "\n".join(out).strip() if not cut else "\n".join(out).strip()

def _fix_typos(text: str) -> str:
    def _sub(m):
        w = m.group(0)
        lower = w.lower()
        repl = _COMMON_TYPO_MAP.get(lower, w)
        return repl.capitalize() if w[0].isupper() else repl
    return _WORD_RE.sub(_sub, text)

def clean_subject_body(subject: str, body: str) -> Tuple[str, str]:
    s = (subject or "").strip()
    b = (body or "").strip()
    if b:
        b = _strip_signature(b)
    s = _fix_typos(s)
    b = _fix_typos(b)
    s = re.sub(r"\s+", " ", s).strip()
    b = re.sub(r"[ \t]+\n", "\n", b)
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    return s, b
