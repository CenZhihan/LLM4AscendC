from __future__ import annotations

import re
from typing import List


def _strip_paths(s: str) -> str:
    s = re.sub(r"/[\w./\-+]+", "<path>", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", s)
    return s


def normalize_anchor(s: str) -> str:
    t = " ".join((s or "").split())[:4000]
    return _strip_paths(t).strip().lower()


def extract_anchor(text: str, max_len: int = 256) -> str:
    """Short stable substring for manifest / comparison."""
    raw = (text or "").strip().replace("\r\n", "\n")
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    pick: List[str] = []
    for ln in lines:
        if any(
            x in ln
            for x in (
                "error:",
                "Error",
                "CMake",
                "undefined reference",
                "fatal:",
                "APIStatusError",
                "correctness",
            )
        ):
            pick.append(ln[:200])
        if sum(len(x) for x in pick) >= max_len:
            break
    blob = " | ".join(pick) if pick else (lines[-1][:200] if lines else raw[:200])
    return blob[:max_len]
