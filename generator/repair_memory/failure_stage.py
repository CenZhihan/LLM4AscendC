from __future__ import annotations

from typing import Any, Dict, Optional


PIPELINE_KEYS = [
    "01-msopgen",
    "02-build",
    "03-install-run",
    "04-pybind-build",
    "05-pybind-install",
    "06-eval",
]


def infer_failure_stage(
    *,
    compiled: Optional[bool],
    correctness: Optional[bool],
    meta_logs: Dict[str, Any],
) -> str:
    """Coarse stage label for tier-B progression."""
    if compiled is True and correctness is True:
        return "success"
    if compiled is True and correctness is not True:
        return "correctness"
    if compiled is not True:
        for key in reversed(PIPELINE_KEYS):
            if str(meta_logs.get(key) or "").strip():
                return key
        return "unknown"
    return "unknown"


_STAGE_RANK_ORDER = [
    "unknown",
    "01-msopgen",
    "02-build",
    "03-install-run",
    "04-pybind-build",
    "05-pybind-install",
    "06-eval",
    "correctness",
    "success",
]


def failure_stage_rank(stage: str) -> int:
    s = (stage or "").strip()
    if s in _STAGE_RANK_ORDER:
        return _STAGE_RANK_ORDER.index(s)
    return 0
