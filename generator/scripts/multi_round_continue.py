"""
Shared helpers for multi-round continue (--continue-attempt) runs.

Used by run_agent_multi_rounds.py (AscendC / MKB ops) and
run_agent_cuda_agent_multi_rounds.py (CUDA-Agent-6K rows).
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

Kind = Literal["ascendc", "cuda"]

_AGGREGATE_PATTERNS: Dict[Kind, re.Pattern[str]] = {
    "ascendc": re.compile(r"^attempts(\d+)_summary_all_ops\.json$"),
    "cuda": re.compile(r"^attempts(\d+)_summary_all_rows\.json$"),
}


@dataclass
class OpContinueState:
    entity_key: str
    action: str  # skip_passed | continue | skip_no_seed
    last_attempt_id: int = 0
    seed_txt_path: Optional[pathlib.Path] = None
    seed_repair_context_path: Optional[pathlib.Path] = None
    prior_op_summary: Dict[str, Any] = field(default_factory=dict)
    prior_attempts: Dict[str, Any] = field(default_factory=dict)
    skip_reason: str = ""
    row_index: Optional[int] = None  # cuda only
    category: str = ""


@dataclass
class ContinuePlan:
    run_dir: pathlib.Path
    continued_from_abs: str
    previous_summary_path: pathlib.Path
    previous_max_attempts: int
    new_max_attempts: int
    previous_aggregate: Dict[str, Any]
    entities: List[OpContinueState]


def aggregate_summary_filename(kind: Kind, max_attempts: int) -> str:
    if kind == "ascendc":
        return f"attempts{max_attempts}_summary_all_ops.json"
    return f"attempts{max_attempts}_summary_all_rows.json"


def per_entity_summary_filename(entity_key: str, max_attempts: int) -> str:
    return f"{entity_key}_attempts{max_attempts}_summary.json"


def find_latest_aggregate_summary(run_dir: pathlib.Path, kind: Kind) -> Tuple[pathlib.Path, int]:
    pattern = _AGGREGATE_PATTERNS[kind]
    best_n = -1
    best_path: Optional[pathlib.Path] = None
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_path = p
    if best_path is None:
        raise FileNotFoundError(
            f"No aggregate summary matching {pattern.pattern!r} under {run_dir}"
        )
    return best_path, best_n


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_keys_from_aggregate(summary: Dict[str, Any], kind: Kind) -> List[str]:
    if kind == "ascendc":
        keys = summary.get("operator_keys")
        if isinstance(keys, list) and keys:
            return [str(k) for k in keys]
        ops = summary.get("ops") or {}
        if isinstance(ops, dict) and ops:
            return list(ops.keys())
        raise ValueError("Aggregate summary missing operator_keys and ops")
    rows = summary.get("rows") or {}
    if isinstance(rows, dict) and rows:
        return list(rows.keys())
    indices = summary.get("resolved_indices") or []
    if indices:
        from tools.cuda_agent_eval.constants import suggested_op_key_ca6k

        return [suggested_op_key_ca6k(int(i)) for i in indices]
    raise ValueError("Aggregate summary missing rows and resolved_indices")


def _entity_section(
    aggregate: Dict[str, Any],
    entity_key: str,
    kind: Kind,
) -> Dict[str, Any]:
    if kind == "ascendc":
        return (aggregate.get("ops") or {}).get(entity_key) or {}
    return (aggregate.get("rows") or {}).get(entity_key) or {}


def _load_per_entity_summary(
    run_dir: pathlib.Path,
    entity_key: str,
    previous_max_attempts: int,
) -> Optional[Dict[str, Any]]:
    path = run_dir / per_entity_summary_filename(entity_key, previous_max_attempts)
    if path.is_file():
        return _load_json(path)
    return None


def _attempt_passed(outcome: Dict[str, Any]) -> bool:
    return outcome.get("compiled") is True and outcome.get("correctness") is True


def _find_last_attempt_id(run_dir: pathlib.Path, entity_key: str) -> int:
    last = 0
    for p in run_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("attempt"):
            continue
        suffix = p.name[len("attempt") :]
        if not suffix.isdigit():
            continue
        n = int(suffix)
        if (p / f"{entity_key}.txt").is_file() and n > last:
            last = n
    return last


def _outcome_for_attempt(section: Dict[str, Any], attempt_id: int) -> Dict[str, Any]:
    attempts = section.get("attempts") or {}
    return dict(attempts.get(f"attempt{attempt_id}") or {})


def _is_passed(
    per_entity: Optional[Dict[str, Any]],
    aggregate_section: Dict[str, Any],
    last_attempt_id: int,
) -> Tuple[bool, Optional[int]]:
    section = per_entity if per_entity is not None else aggregate_section
    fixed = section.get("fixed_on_attempt")
    if fixed is not None:
        return True, int(fixed)
    if last_attempt_id <= 0:
        return False, None
    outcome = _outcome_for_attempt(section, last_attempt_id)
    if outcome and _attempt_passed(outcome):
        return True, last_attempt_id
    return False, None


def _rebuild_repair_context(
    *,
    entity_key: str,
    eval_result_path: str,
    max_log_lines: int,
    kind: Kind,
) -> Optional[str]:
    path = pathlib.Path(eval_result_path)
    if not path.is_file():
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    from generator.repair_memory.error_signals import select_error_log_paths

    if kind == "ascendc":
        from generator.scripts.run_agent_multi_rounds import (
            _build_repair_error_context,
            _extract_eval_core,
        )

        core = _extract_eval_core(payload, entity_key)
        selected = select_error_log_paths(core["logs"])
        return _build_repair_error_context(
            op=entity_key,
            result_payload=payload,
            selected_logs=selected,
            max_log_lines=max_log_lines,
        )
    from generator.scripts.run_agent_cuda_agent_multi_rounds import (
        _build_repair_error_context,
        _extract_eval_core,
    )

    core = _extract_eval_core(payload, entity_key)
    selected = select_error_log_paths(core["logs"])
    return _build_repair_error_context(
        op=entity_key,
        result_payload=payload,
        selected_logs=selected,
        max_log_lines=max_log_lines,
    )


def _ensure_seed_repair_context(
    *,
    run_dir: pathlib.Path,
    entity_key: str,
    last_attempt_id: int,
    repair_path: pathlib.Path,
    section: Dict[str, Any],
    max_log_lines: int,
    kind: Kind,
) -> Tuple[Optional[pathlib.Path], str]:
    if repair_path.is_file():
        return repair_path, ""
    if last_attempt_id < 2:
        return repair_path, ""
    outcome = _outcome_for_attempt(section, last_attempt_id)
    eval_path = str(outcome.get("eval_result_path") or "")
    if not eval_path:
        return None, "missing repair_context and eval_result_path in summary"
    text = _rebuild_repair_context(
        entity_key=entity_key,
        eval_result_path=eval_path,
        max_log_lines=max_log_lines,
        kind=kind,
    )
    if not text:
        return None, "failed to rebuild repair_context from eval json"
    repair_path.parent.mkdir(parents=True, exist_ok=True)
    repair_path.write_text(text, encoding="utf-8")
    return repair_path, "rebuilt_repair_context"


def build_continue_plan(
    *,
    run_dir: pathlib.Path,
    new_max_attempts: int,
    kind: Kind,
    ops_filter: Optional[List[str]] = None,
    max_log_lines: int = 220,
) -> ContinuePlan:
    run_dir = run_dir.resolve()
    summary_path, previous_max = find_latest_aggregate_summary(run_dir, kind)
    if new_max_attempts <= previous_max:
        raise ValueError(
            f"--max-attempts ({new_max_attempts}) must be > previous max ({previous_max}) "
            f"from {summary_path.name}"
        )
    aggregate = _load_json(summary_path)
    all_keys = _entity_keys_from_aggregate(aggregate, kind)
    if ops_filter is not None:
        unknown = [k for k in ops_filter if k not in all_keys]
        if unknown:
            raise ValueError(f"--ops not present in source run: {unknown!r}")
        entity_keys = [k for k in all_keys if k in ops_filter]
    else:
        entity_keys = all_keys

    entities: List[OpContinueState] = []
    for entity_key in entity_keys:
        per_entity = _load_per_entity_summary(run_dir, entity_key, previous_max)
        agg_section = _entity_section(aggregate, entity_key, kind)
        section = per_entity if per_entity is not None else agg_section
        prior_attempts = dict(section.get("attempts") or {})
        prior_op_summary = dict(section)
        category = str(
            section.get("category")
            or agg_section.get("category")
            or "activation"
        )
        row_index: Optional[int] = None
        if kind == "cuda":
            ri = section.get("row_index") or agg_section.get("row_index")
            if ri is not None:
                row_index = int(ri)

        last_id = _find_last_attempt_id(run_dir, entity_key)
        passed, fixed_on = _is_passed(per_entity, agg_section, last_id)
        if passed:
            entities.append(
                OpContinueState(
                    entity_key=entity_key,
                    action="skip_passed",
                    last_attempt_id=last_id or (fixed_on or 0),
                    prior_op_summary=prior_op_summary,
                    prior_attempts=prior_attempts,
                    skip_reason=f"fixed_on_attempt={fixed_on}",
                    row_index=row_index,
                    category=category,
                )
            )
            continue

        if last_id <= 0:
            entities.append(
                OpContinueState(
                    entity_key=entity_key,
                    action="skip_no_seed",
                    prior_op_summary=prior_op_summary,
                    prior_attempts=prior_attempts,
                    skip_reason="no attempt directory with entity txt found",
                    row_index=row_index,
                    category=category,
                )
            )
            continue

        start_id = last_id + 1
        if start_id > new_max_attempts:
            entities.append(
                OpContinueState(
                    entity_key=entity_key,
                    action="skip_passed",
                    last_attempt_id=last_id,
                    prior_op_summary=prior_op_summary,
                    prior_attempts=prior_attempts,
                    skip_reason=f"last_attempt_id={last_id} already at or beyond max_attempts={new_max_attempts}",
                    row_index=row_index,
                    category=category,
                )
            )
            continue

        seed_txt = run_dir / f"attempt{last_id}" / f"{entity_key}.txt"
        seed_repair = run_dir / f"attempt{last_id}" / f"{entity_key}_repair_context.txt"
        repair_path, repair_note = _ensure_seed_repair_context(
            run_dir=run_dir,
            entity_key=entity_key,
            last_attempt_id=last_id,
            repair_path=seed_repair,
            section=section,
            max_log_lines=max_log_lines,
            kind=kind,
        )
        if last_id >= 2 and repair_path is None:
            entities.append(
                OpContinueState(
                    entity_key=entity_key,
                    action="skip_no_seed",
                    last_attempt_id=last_id,
                    seed_txt_path=seed_txt,
                    prior_op_summary=prior_op_summary,
                    prior_attempts=prior_attempts,
                    skip_reason=repair_note or "missing repair_context",
                    row_index=row_index,
                    category=category,
                )
            )
            continue

        entities.append(
            OpContinueState(
                entity_key=entity_key,
                action="continue",
                last_attempt_id=last_id,
                seed_txt_path=seed_txt,
                seed_repair_context_path=repair_path,
                prior_op_summary=prior_op_summary,
                prior_attempts=prior_attempts,
                skip_reason=repair_note,
                row_index=row_index,
                category=category,
            )
        )

    return ContinuePlan(
        run_dir=run_dir,
        continued_from_abs=str(run_dir),
        previous_summary_path=summary_path,
        previous_max_attempts=previous_max,
        new_max_attempts=new_max_attempts,
        previous_aggregate=aggregate,
        entities=entities,
    )


def merge_op_summary(
    prior: Dict[str, Any],
    new_partial: Dict[str, Any],
    *,
    new_max_attempts: int,
    continued_from_abs: str,
    source_last_attempt: int,
    continue_session_at: str,
    ran_new_attempts: bool,
) -> Dict[str, Any]:
    merged_attempts = dict(prior.get("attempts") or {})
    merged_attempts.update(new_partial.get("attempts") or {})
    out = dict(prior)
    out.update(new_partial)
    out["attempts"] = merged_attempts
    out["max_attempts"] = new_max_attempts
    if ran_new_attempts:
        if new_partial.get("fixed_on_attempt") is not None:
            out["fixed_on_attempt"] = new_partial.get("fixed_on_attempt")
            out["final_status"] = new_partial.get("final_status", out.get("final_status"))
            fo = new_partial.get("fixed_on_attempt")
            out["fixed_in_attempt2"] = fo == 2 if fo is not None else False
        elif new_partial.get("final_status"):
            out["final_status"] = new_partial["final_status"]
    out["continue_meta"] = {
        "continued_from": continued_from_abs,
        "source_last_attempt": source_last_attempt,
        "continue_session_at": continue_session_at,
    }
    return out


def write_continue_report(
    *,
    run_dir: pathlib.Path,
    plan: ContinuePlan,
    continue_session_at: str,
    model: str,
    use_repair_memory: bool,
    entity_results: Dict[str, Dict[str, Any]],
) -> pathlib.Path:
    stats = {"continued": 0, "skipped_passed": 0, "skipped_no_seed": 0}
    entities_out: Dict[str, Any] = {}
    for ent in plan.entities:
        key = ent.entity_key
        rec: Dict[str, Any] = {
            "action": ent.action,
            "last_attempt_id": ent.last_attempt_id,
        }
        if ent.action == "skip_passed":
            stats["skipped_passed"] += 1
            prior = ent.prior_op_summary
            if prior.get("fixed_on_attempt") is not None:
                rec["fixed_on_attempt"] = prior["fixed_on_attempt"]
        elif ent.action == "skip_no_seed":
            stats["skipped_no_seed"] += 1
            rec["reason"] = ent.skip_reason
        elif ent.action == "continue":
            stats["continued"] += 1
            rec["new_attempt_range"] = [ent.last_attempt_id + 1, plan.new_max_attempts]
            if ent.seed_txt_path:
                rec["seed_txt"] = str(ent.seed_txt_path)
            if ent.seed_repair_context_path:
                rec["seed_repair_context"] = str(ent.seed_repair_context_path)
            if ent.skip_reason:
                rec["seed_note"] = ent.skip_reason
            result = entity_results.get(key) or {}
            if result.get("fixed_on_attempt") is not None:
                rec["outcome_fixed_on_attempt"] = result["fixed_on_attempt"]
            rec["outcome_final_status"] = result.get("final_status", "")
        entities_out[key] = rec

    ts = continue_session_at.replace(":", "").replace("-", "")
    fname = f"continue_report_{ts}.json"
    report_path = run_dir / fname
    report = {
        "continued_from": plan.continued_from_abs,
        "continue_started_at": continue_session_at,
        "previous_max_attempts": plan.previous_max_attempts,
        "new_max_attempts": plan.new_max_attempts,
        "model": model,
        "use_repair_memory": use_repair_memory,
        "entities": entities_out,
        "stats": stats,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
