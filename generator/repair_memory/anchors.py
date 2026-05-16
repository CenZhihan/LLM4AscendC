from __future__ import annotations

import re
from typing import List

from tools.common.error_extract import anchor_from_excerpt, parse_layered_correctness_info


def _strip_paths(s: str) -> str:
    s = re.sub(r"/[\w./\-+]+", "<path>", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", s)
    return s


def normalize_anchor(s: str) -> str:
    t = " ".join((s or "").split())[:4000]
    return _strip_paths(t).strip().lower()


def extract_anchor(text: str, max_len: int = 256) -> str:
    """Short stable substring for manifest / comparison (prefers root_cause when layered)."""
    root, symptom = parse_layered_correctness_info(text or "")
    if root.strip():
        a = anchor_from_excerpt(root, max_len=max_len)
        if a:
            return a
    return anchor_from_excerpt(symptom or text, max_len=max_len)
