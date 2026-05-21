"""Machine-readable JSON + human delimiter block for advisory tools."""
from __future__ import annotations

import json
from typing import Any, Dict


ADVISORY_SEPARATOR = "#######"


def _nl_summary(payload: Dict[str, Any]) -> str:
    tool = payload.get("tool", "advisory")
    summ = payload.get("summary", "")
    lines = [f"[{tool}] {summ}", f"CANN baseline: {payload.get('cann_version', '')}"]
    conf = payload.get("alignment_confidence")
    if conf:
        lines.append(f"Confidence: {conf}")
    pw = payload.get("parse_warnings") or []
    if pw:
        lines.append("Warnings: " + "; ".join(str(x) for x in pw[:5]))
    recs = payload.get("recommendations") or []
    if recs:
        lines.append("Top recommendations:")
        for r in recs[:6]:
            lines.append(f"- {r}")
    return "\n".join(lines)


def advisory_display_string(payload: Dict[str, Any]) -> str:
    """
    Return ``json.dumps(payload) + ####### + short natural language``.

    The prefix is parseable JSON for downstream tooling; text after the delimiter is explanatory only.
    """
    body = json.dumps(payload, ensure_ascii=False, indent=None)
    return f"{body}\n{ADVISORY_SEPARATOR}\n{_nl_summary(payload)}"

