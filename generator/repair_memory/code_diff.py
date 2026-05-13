from __future__ import annotations

import difflib
from typing import List


def format_attempt_code_diff(
    prev_code: str,
    curr_code: str,
    *,
    prev_label: str = "previous_attempt",
    curr_label: str = "current_attempt",
    max_chars: int = 12000,
) -> str:
    """
    Bounded unified diff between two attempt txt bodies for the review LLM.

    If outputs would exceed max_chars, keep the prefix and append a truncation notice.
    """
    a_lines: List[str] = (prev_code or "").splitlines()
    b_lines: List[str] = (curr_code or "").splitlines()
    if not a_lines and not b_lines:
        return "(both code snapshots empty)"
    diff_lines = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=prev_label,
            tofile=curr_label,
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no line-level differences; files may differ only in line endings or are identical)"
    text = "\n".join(diff_lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n\n... (unified diff truncated; focus on the earliest hunks above)\n"
