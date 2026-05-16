from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .code_diff import format_attempt_code_diff
from .error_signals import build_attempt_error_bundle, format_attempt_signals_for_review
from .failure_stage import infer_failure_stage
from .inbox import write_memory_inbox_line
from .merge import merge_run_inbox
from .paths import get_memory_root, is_repair_memory_enabled
from .review_llm import generate_repair_natural_language
from .schema import RepairMemoryRecord, SCHEMA_VERSION
from .tier_gate import classify_tier_and_gates, code_digest


def _infra_skip(text: str) -> bool:
    t = text or ""
    return "Error code: 402" in t and "APIStatusError" in t


def maybe_write_repair_memory_after_eval(
    *,
    op: str,
    category: str,
    strategy: str,
    tool_mode: str,
    eval_mode: str,
    attempt_id: int,
    run_dir: Path,
    run_slug: str,
    llm_config: Dict[str, Any],
    op_summary_attempts: Dict[str, Any],
    curr_outcome: Dict[str, Any],
    curr_payload: Dict[str, Any],
    prev_code: str,
    curr_code: str,
    memory_root: Optional[Path] = None,
    max_log_lines: int = 220,
) -> None:
    if not is_repair_memory_enabled():
        return
    if attempt_id < 2:
        return
    prev = op_summary_attempts.get(f"attempt{attempt_id - 1}")
    if not isinstance(prev, dict):
        return
    prev_path = (prev.get("eval_result_path") or "").strip()
    if not prev_path:
        return
    ppath = Path(prev_path)
    if not ppath.is_file():
        return
    try:
        prev_payload = json.loads(ppath.read_text(encoding="utf-8"))
    except Exception:
        return

    pc = (prev_payload.get("result") or {}).get(op) or {}
    cc = (curr_payload.get("result") or {}).get(op) or {}
    info = f"{pc.get('correctness_info','')}\n{cc.get('correctness_info','')}"
    if _infra_skip(info):
        return

    tier, _reason = classify_tier_and_gates(
        op=op,
        prev_outcome=prev,
        curr_outcome=curr_outcome,
        prev_payload=prev_payload,
        curr_payload=curr_payload,
        prev_code=prev_code,
        curr_code=curr_code,
    )
    if tier is None:
        return

    meta_p = (prev_payload.get("meta") or {})
    meta_c = (curr_payload.get("meta") or {})
    stage_b = infer_failure_stage(
        compiled=pc.get("compiled"),
        correctness=pc.get("correctness"),
        meta_logs=meta_p.get("logs") or {},
    )
    stage_a = infer_failure_stage(
        compiled=cc.get("compiled"),
        correctness=cc.get("correctness"),
        meta_logs=meta_c.get("logs") or {},
    )

    bundle_prev = build_attempt_error_bundle(prev_payload, op, max_log_lines=max_log_lines)
    bundle_curr = build_attempt_error_bundle(curr_payload, op, max_log_lines=max_log_lines)

    prev_summary = format_attempt_signals_for_review(bundle_prev, failure_stage=stage_b)
    curr_summary = format_attempt_signals_for_review(bundle_curr, failure_stage=stage_a)

    code_diff_text = format_attempt_code_diff(
        prev_code or "",
        curr_code or "",
        prev_label=f"attempt{attempt_id - 1}_txt",
        curr_label=f"attempt{attempt_id}_txt",
        max_chars=12000,
    )

    nl = generate_repair_natural_language(
        llm_config=llm_config,
        tier=tier,
        prev_summary=prev_summary,
        curr_summary=curr_summary,
        code_diff_text=code_diff_text,
    )
    if not nl.strip():
        return

    mem_id = RepairMemoryRecord.new_id()
    transition: Dict[str, Any] = {
        "compiled": [pc.get("compiled"), cc.get("compiled")],
        "correctness": [pc.get("correctness"), cc.get("correctness")],
    }
    evidence_refs = [
        str(run_dir / f"attempt{attempt_id - 1}" / f"{op}.txt"),
        str(run_dir / f"attempt{attempt_id}" / f"{op}.txt"),
        prev_path,
        str(curr_outcome.get("eval_result_path") or ""),
    ]
    rec = RepairMemoryRecord(
        memory_id=mem_id,
        schema_version=SCHEMA_VERSION,
        tier=tier,
        confidence="high" if tier == "A" else "medium",
        op_key=op,
        category=category,
        tool_mode=tool_mode,
        strategy=strategy,
        eval_mode=eval_mode,
        transition=transition,
        failure_stage_before=stage_b,
        failure_stage_after=stage_a,
        error_anchors_before=bundle_prev.symptom_anchor,
        error_anchors_after=bundle_curr.symptom_anchor,
        symptom_anchor_before=bundle_prev.symptom_anchor,
        symptom_anchor_after=bundle_curr.symptom_anchor,
        root_cause_anchor_before=bundle_prev.root_cause_anchor,
        root_cause_anchor_after=bundle_curr.root_cause_anchor,
        code_digest_before=code_digest(prev_code),
        code_digest_after=code_digest(curr_code),
        natural_language=nl,
        evidence_refs=evidence_refs,
    )
    d = rec.to_json_dict()

    root = memory_root if memory_root is not None else get_memory_root()
    write_memory_inbox_line(d, run_slug=run_slug)
    merge_run_inbox(root, run_slug)
