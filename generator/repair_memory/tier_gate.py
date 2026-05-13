from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

from .anchors import extract_anchor, normalize_anchor
from .failure_stage import failure_stage_rank, infer_failure_stage


def code_digest(code: str) -> str:
    raw = (code or "").encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def _extract_core(payload: Dict[str, Any], op: str) -> Dict[str, Any]:
    result = (payload.get("result") or {}).get(op) or {}
    meta = payload.get("meta") or {}
    return {
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "correctness_info": result.get("correctness_info") or "",
        "logs": meta.get("logs") or {},
    }


def classify_tier_and_gates(
    *,
    op: str,
    prev_outcome: Dict[str, Any],
    curr_outcome: Dict[str, Any],
    prev_payload: Dict[str, Any],
    curr_payload: Dict[str, Any],
    prev_code: str,
    curr_code: str,
) -> Tuple[Optional[str], str]:
    """
    Returns (tier or None if should not write, reason string).
    tier is 'A' or 'B'.
    """
    if curr_outcome.get("eval_ran") is not True:
        return None, "curr_eval_not_ran"
    if prev_outcome.get("eval_ran") is not True:
        return None, "prev_eval_not_ran"

    pc = _extract_core(prev_payload, op)
    cc = _extract_core(curr_payload, op)

    pb = pc.get("compiled")
    pq = pc.get("correctness")
    cb = cc.get("compiled")
    cq = cc.get("correctness")

    if pb is None or cb is None:
        return None, "missing_compiled_flag"

    dig_p = code_digest(prev_code)
    dig_c = code_digest(curr_code)
    if dig_p == dig_c:
        return None, "no_code_change"

    stage_b = infer_failure_stage(compiled=pb, correctness=pq, meta_logs=pc["logs"])
    stage_a = infer_failure_stage(compiled=cb, correctness=cq, meta_logs=cc["logs"])

    anchor_b = extract_anchor(pc.get("correctness_info", ""))
    anchor_a = extract_anchor(cc.get("correctness_info", ""))

    # Tier A: objective repair signal
    compile_fixed = (pb is not True) and (cb is True)
    value_fixed = (pb is True) and (pq is not True) and (cq is True)
    if compile_fixed or value_fixed:
        return "A", "tier_a_objective_gain"

    # Still failing overall?
    still_bad = (cb is not True) or (cq is not True)
    if not still_bad:
        return None, "already_pass_no_memory"

    # Tier B: stage rank increases, anchors differ (normalized)
    rb = failure_stage_rank(stage_b)
    ra = failure_stage_rank(stage_a)
    if ra <= rb:
        return None, "stage_rank_not_progressing"
    if stage_b == stage_a:
        return None, "stage_label_unchanged"
    if normalize_anchor(anchor_b) == normalize_anchor(anchor_a):
        return None, "anchors_normalized_same"

    return "B", "tier_b_stage_and_anchor_progress"
