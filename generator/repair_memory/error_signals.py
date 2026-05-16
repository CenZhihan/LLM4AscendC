"""Build structured error bundles from eval result payloads (shared by repair context + memory write)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tools.common.error_extract import (
    anchor_from_excerpt,
    build_layered_errors_from_log_text,
    parse_layered_correctness_info,
    read_log_tail_text,
)

PREFERRED_LOG_KEYS = ["02-build", "06-eval"]
FALLBACK_LOG_KEYS = ["01-msopgen", "03-install-run", "04-pybind-build", "05-pybind-install"]


def select_error_log_paths(logs: Dict[str, str]) -> List[str]:
    selected: List[str] = []
    for key in PREFERRED_LOG_KEYS:
        value = (logs.get(key) or "").strip()
        if value:
            selected.append(value)
    if selected:
        return selected
    for key in FALLBACK_LOG_KEYS:
        value = (logs.get(key) or "").strip()
        if value:
            selected.append(value)
    return selected


def read_log_file_tail(path: str | Path, max_lines: int) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return read_log_tail_text(raw, max_lines)


def concat_log_excerpts(paths: List[str], max_lines: int) -> str:
    parts: List[str] = []
    for p in paths:
        excerpt = read_log_file_tail(p, max_lines)
        if excerpt:
            parts.append(f"=== log file: {p} (tail {max_lines} lines) ===\n{excerpt}")
    return "\n\n".join(parts)


@dataclass
class AttemptErrorBundle:
    compiled: Any
    correctness: Any
    correctness_info: str
    root_cause: str
    symptom: str
    root_cause_anchor: str
    symptom_anchor: str
    log_excerpt: str
    selected_log_paths: List[str]


def _extract_eval_core(payload: Dict[str, Any], op: str) -> Dict[str, Any]:
    result = (payload.get("result") or {}).get(op) or {}
    meta = payload.get("meta") or {}
    return {
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "correctness_info": result.get("correctness_info") or "",
        "logs": meta.get("logs") or {},
    }


def build_attempt_error_bundle(
    payload: Dict[str, Any],
    op: str,
    *,
    max_log_lines: int = 220,
) -> AttemptErrorBundle:
    core = _extract_eval_core(payload, op)
    logs: Dict[str, str] = core["logs"]
    selected = select_error_log_paths(logs)
    log_excerpt = concat_log_excerpts(selected, max_log_lines)

    ci = core["correctness_info"]
    root, symptom = parse_layered_correctness_info(ci)
    if not root and not symptom and log_excerpt:
        root, symptom, ci = build_layered_errors_from_log_text(log_excerpt)
    elif log_excerpt and not root:
        root = build_layered_errors_from_log_text(log_excerpt)[0]

    root_anchor = anchor_from_excerpt(root) if root else ""
    symptom_anchor = anchor_from_excerpt(symptom or ci)

    return AttemptErrorBundle(
        compiled=core["compiled"],
        correctness=core["correctness"],
        correctness_info=ci,
        root_cause=root,
        symptom=symptom,
        root_cause_anchor=root_anchor,
        symptom_anchor=symptom_anchor,
        log_excerpt=log_excerpt,
        selected_log_paths=selected,
    )


def format_attempt_signals_for_review(
    bundle: AttemptErrorBundle,
    *,
    failure_stage: str,
    max_chars: int = 6000,
) -> str:
    parts = [
        f"compiled={bundle.compiled} correctness={bundle.correctness}",
        f"failure_stage={failure_stage}",
        f"root_cause_anchor={bundle.root_cause_anchor}",
        f"symptom_anchor={bundle.symptom_anchor}",
        "",
        "correctness_info:",
        bundle.correctness_info or "(empty)",
        "",
        "log_excerpt:",
        bundle.log_excerpt or "(no log files)",
    ]
    text = "\n".join(parts)
    if len(text) > max_chars:
        # Prefer keeping root_cause block and start of log
        head = "\n".join(parts[:8])
        budget = max_chars - len(head) - 20
        tail = (bundle.log_excerpt or "")[-max(0, budget) :]
        text = head + "\n...(truncated log_excerpt tail)...\n" + tail
    return text


def format_repair_error_context(
    *,
    op: str,
    bundle: AttemptErrorBundle,
    attempt_label: str = "attempt",
) -> str:
    sections: List[str] = [
        f"=== {attempt_label} eval summary for {op} ===",
        f"compiled: {bundle.compiled}",
        f"correctness: {bundle.correctness}",
        "",
    ]
    if bundle.correctness_info:
        sections.append("=== correctness_info (layered) ===")
        sections.append(bundle.correctness_info)
        sections.append("")
    if bundle.root_cause:
        sections.append("=== root_cause (excerpt) ===")
        sections.append(bundle.root_cause)
        sections.append("")
    if bundle.symptom:
        sections.append("=== symptom (excerpt) ===")
        sections.append(bundle.symptom)
        sections.append("")
    if bundle.log_excerpt:
        sections.append(bundle.log_excerpt)
        sections.append("")
    return "\n".join(sections).strip() + "\n"
