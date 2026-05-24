#!/usr/bin/env python3
"""
Multi-attempt agent generation + eval + repair for CUDA-Agent-Ops-6K rows.

Separate entry from run_agent_multi_rounds.py (MKB single-op). Tasks are keyed ca6k_{row:05d}.
Eval invokes tools/eval_cuda_agent_operator.py with --dataset-path + --row-index.

Environment (inherited by subprocess): USE_API_CONFIG, LLM4ASCENDC_REF_ON_CPU,
LLM4ASCENDC_CUDA_AGENT_ART_ROOT, LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH, etc.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "output"


@dataclass
class AttemptOutcome:
    attempt_id: int
    generated: bool
    generation_error: str
    txt_path: str
    report_path: str
    eval_ran: bool
    eval_result_path: str
    compiled: Optional[bool]
    correctness: Optional[bool]
    correctness_info: str
    selected_log_paths: List[str]
    repair_context_path: str


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _normalize_ascend_search_version(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    t = str(value).strip()
    return t or None


def _artifact_group_rel_from_txt_path(txt_path: pathlib.Path) -> Optional[pathlib.Path]:
    txt = txt_path.resolve()
    out_root = OUTPUT_ROOT.resolve()
    try:
        rel_parent = txt.parent.relative_to(out_root)
    except Exception:
        return None
    if str(rel_parent) in ("", "."):
        return None
    return rel_parent


def _cuda_agent_eval_result_json_path(txt_path: pathlib.Path, op_key: str) -> pathlib.Path:
    from tools.cuda_agent_eval.constants import default_cuda_agent_art_root

    rel_group = _artifact_group_rel_from_txt_path(txt_path)
    art_root = default_cuda_agent_art_root(REPO_ROOT)
    if rel_group is not None:
        art_root = art_root / rel_group
    return art_root / op_key / f"result_{op_key}.json"


def _run_eval_cuda_agent_for_txt(
    txt_path: pathlib.Path,
    dataset_path: pathlib.Path,
    row_index: int,
    mode: str,
    clean_policy: str,
    eval_workers: int = 1,
    eval_npu: int = 1,
) -> int:
    if eval_workers != 1:
        raise ValueError(
            "Per-attempt eval currently supports --eval-workers=1 only; "
            "use op-level parallelism via --parallel-ops."
        )
    _ = eval_npu
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "eval_cuda_agent_operator.py"),
        "--txt",
        str(txt_path),
        "--dataset-path",
        str(dataset_path),
        "--row-index",
        str(row_index),
        "--mode",
        mode,
        "--clean-policy",
        clean_policy,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, env=os.environ.copy())
    return int(proc.returncode)


def _load_result_payload(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_result_payload_when_eval_json_missing(
    *,
    op: str,
    eval_mode: str,
    eval_rc: int,
    result_path: pathlib.Path,
) -> Dict[str, Any]:
    msg = (
        f"eval_cuda_agent_operator did not emit result json (subprocess rc={eval_rc}). "
        f"Expected path: {result_path}"
    )
    return {
        "result": {
            op: {
                "compiled": False,
                "correctness": False,
                "performance": None,
                "correctness_info": msg,
            }
        },
        "meta": {"mode": eval_mode, "fingerprint": None, "logs": {}},
    }


def _extract_eval_core(payload: Dict[str, Any], op: str) -> Dict[str, Any]:
    result = (payload.get("result") or {}).get(op) or {}
    meta = payload.get("meta") or {}
    return {
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "correctness_info": result.get("correctness_info") or "",
        "logs": meta.get("logs") or {},
    }


def _build_repair_error_context(
    *,
    op: str,
    result_payload: Dict[str, Any],
    selected_logs: List[str],
    max_log_lines: int,
) -> str:
    from generator.repair_memory.error_signals import (
        build_attempt_error_bundle,
        format_repair_error_context,
    )

    bundle = build_attempt_error_bundle(result_payload, op, max_log_lines=max_log_lines)
    return format_repair_error_context(op=op, bundle=bundle, attempt_label="prior attempt")


def _save_generation_outputs(
    *,
    out_dir: pathlib.Path,
    op: str,
    generated_code: str,
    report: Optional[Dict[str, Any]],
    reasoning: Optional[str],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{op}.txt"
    txt_path.write_text(generated_code or "", encoding="utf-8")
    report_path = out_dir / f"{op}_report.json"
    if report is not None:
        _write_json(report_path, report)
    else:
        _write_json(report_path, {})
    if reasoning:
        _write_text(out_dir / f"{op}_cot.txt", reasoning)
    return {"txt_path": str(txt_path), "report_path": str(report_path)}


def _generate_one_cuda_attempt(
    *,
    op_key: str,
    row_index: int,
    row: Dict[str, Any],
    category: str,
    strategy: str,
    tool_mode: str,
    llm_config: Dict[str, Any],
    out_dir: pathlib.Path,
    attempt_id: int,
    repair_error_logs_raw: str = "",
    previous_attempt_code: str = "",
    ascend_search_version_filter: Optional[str] = None,
    retrieved_repair_memories: str = "",
    retrieved_repair_memories_applied: Optional[List[Dict[str, Any]]] = None,
    repair_memory_selection: Optional[Dict[str, Any]] = None,
    eval_mode: str = "full",
) -> Dict[str, Any]:
    from generator.agent.agent_runner import KernelGenerationTask, generate_kernel_with_agent

    task = KernelGenerationTask(
        language="ascendc",
        op=op_key,
        strategy_name=strategy,
        category=category,
        cuda_agent_row=row,
        cuda_agent_row_index=row_index,
    )
    result = generate_kernel_with_agent(
        task,
        tool_mode=tool_mode,
        llm_config=llm_config,
        attempt_id=attempt_id,
        repair_error_logs_raw=repair_error_logs_raw,
        previous_attempt_code=previous_attempt_code,
        ascend_search_version_filter=ascend_search_version_filter,
        retrieved_repair_memories=retrieved_repair_memories,
        retrieved_repair_memories_applied=retrieved_repair_memories_applied,
        repair_memory_selection=repair_memory_selection,
        eval_mode=eval_mode,
    )
    paths = _save_generation_outputs(
        out_dir=out_dir,
        op=op_key,
        generated_code=result.generated_code,
        report=result.report,
        reasoning=result.reasoning,
    )
    return {
        "result": result,
        "txt_path": pathlib.Path(paths["txt_path"]),
        "report_path": pathlib.Path(paths["report_path"]),
    }


def _attempt_failed(compiled: Optional[bool], correctness: Optional[bool]) -> bool:
    if compiled is not True:
        return True
    return correctness is not True


def run_multi_attempt_for_cuda_row(
    *,
    op_key: str,
    row_index: int,
    row: Dict[str, Any],
    dataset_path: pathlib.Path,
    category: str,
    strategy: str,
    tool_mode: str,
    llm_config: Dict[str, Any],
    run_dir: pathlib.Path,
    eval_mode: str,
    clean_policy: str,
    max_log_lines: int,
    max_attempts: int,
    eval_workers: int,
    eval_npu: int,
    ascend_search_version_filter: Optional[str] = None,
    run_slug: str = "",
    memory_root: Optional[pathlib.Path] = None,
    use_repair_memory: bool = False,
    start_attempt_id: int = 1,
    seed_previous_code: str = "",
    seed_repair_context_path: str = "",
    prior_attempts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if max_attempts < 2:
        raise ValueError("max_attempts must be >= 2")
    if start_attempt_id < 1 or start_attempt_id > max_attempts:
        raise ValueError(f"start_attempt_id must be in [1, {max_attempts}], got {start_attempt_id}")
    if not (run_slug or "").strip():
        from generator.repair_memory.paths import run_slug_from_run_dir

        run_slug = run_slug_from_run_dir(run_dir) if use_repair_memory else ""
    op_summary: Dict[str, Any] = {
        "op_key": op_key,
        "row_index": row_index,
        "category": category,
        "attempts": dict(prior_attempts or {}),
        "fixed_in_attempt2": False,
        "fixed_on_attempt": None,
        "max_attempts": max_attempts,
        "final_status": "unknown",
    }
    previous_attempt_code = seed_previous_code
    previous_repair_context_path = seed_repair_context_path

    for attempt_id in range(start_attempt_id, max_attempts + 1):
        attempt_dir = run_dir / f"attempt{attempt_id}"
        outcome = AttemptOutcome(
            attempt_id=attempt_id,
            generated=False,
            generation_error="",
            txt_path="",
            report_path="",
            eval_ran=False,
            eval_result_path="",
            compiled=None,
            correctness=None,
            correctness_info="",
            selected_log_paths=[],
            repair_context_path="",
        )
        try:
            saved_prev_for_memory = previous_attempt_code
            repair_logs_raw = ""
            if attempt_id >= 2:
                if not previous_repair_context_path:
                    raise RuntimeError(f"attempt{attempt_id} missing previous repair context")
                repair_logs_raw = pathlib.Path(previous_repair_context_path).read_text(
                    encoding="utf-8", errors="replace"
                )
            retrieved_repair_memories = ""
            retrieved_repair_memories_applied: List[Dict[str, Any]] = []
            repair_memory_selection: Optional[Dict[str, Any]] = None
            if use_repair_memory:
                try:
                    from generator.repair_memory import build_retrieval_block_for_attempt

                    (
                        retrieved_repair_memories,
                        retrieved_repair_memories_applied,
                        repair_memory_selection,
                    ) = build_retrieval_block_for_attempt(
                        llm_config=llm_config,
                        op=op_key,
                        category=category,
                        tool_mode=tool_mode,
                        eval_mode=eval_mode,
                        repair_error_logs_raw=repair_logs_raw,
                        attempt_id=attempt_id,
                        memory_root=memory_root,
                    )
                except Exception as e:
                    retrieved_repair_memories = ""
                    retrieved_repair_memories_applied = []
                    repair_memory_selection = {
                        "memory_ids": [],
                        "memory_ids_resolved": [],
                        "memory_ids_dropped": [],
                        "selection_rationale": (
                            f"build_retrieval_block_for_attempt raised: {e!s}"
                        ),
                        "raw_model_output": "",
                        "parse_ok": False,
                        "parse_error": repr(e),
                    }
            gen = _generate_one_cuda_attempt(
                op_key=op_key,
                row_index=row_index,
                row=row,
                category=category,
                strategy=strategy,
                tool_mode=tool_mode,
                llm_config=llm_config,
                out_dir=attempt_dir,
                attempt_id=attempt_id,
                repair_error_logs_raw=repair_logs_raw,
                previous_attempt_code=previous_attempt_code,
                ascend_search_version_filter=ascend_search_version_filter,
                retrieved_repair_memories=retrieved_repair_memories,
                retrieved_repair_memories_applied=retrieved_repair_memories_applied,
                repair_memory_selection=repair_memory_selection,
                eval_mode=eval_mode,
            )
            outcome.generated = True
            outcome.txt_path = str(gen["txt_path"])
            outcome.report_path = str(gen["report_path"])

            eval_rc = _run_eval_cuda_agent_for_txt(
                gen["txt_path"],
                dataset_path=dataset_path,
                row_index=row_index,
                mode=eval_mode,
                clean_policy=clean_policy,
                eval_workers=eval_workers,
                eval_npu=eval_npu,
            )
            outcome.eval_ran = True
            result_path = _cuda_agent_eval_result_json_path(gen["txt_path"], op_key)
            if not result_path.exists():
                payload = _synthetic_result_payload_when_eval_json_missing(
                    op=op_key,
                    eval_mode=eval_mode,
                    eval_rc=eval_rc,
                    result_path=result_path,
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(result_path, payload)
            else:
                payload = _load_result_payload(result_path)
            outcome.eval_result_path = str(result_path)
            core = _extract_eval_core(payload, op_key)
            outcome.compiled = core["compiled"]
            outcome.correctness = core["correctness"]
            outcome.correctness_info = core["correctness_info"]
            from generator.repair_memory.error_signals import select_error_log_paths

            outcome.selected_log_paths = select_error_log_paths(core["logs"])

            repair_text = _build_repair_error_context(
                op=op_key,
                result_payload=payload,
                selected_logs=outcome.selected_log_paths,
                max_log_lines=max_log_lines,
            )
            repair_path = attempt_dir / f"{op_key}_repair_context.txt"
            _write_text(repair_path, repair_text)
            outcome.repair_context_path = str(repair_path)

            if use_repair_memory:
                try:
                    from generator.repair_memory import maybe_write_repair_memory_after_eval

                    maybe_write_repair_memory_after_eval(
                        op=op_key,
                        category=category,
                        strategy=strategy,
                        tool_mode=tool_mode,
                        eval_mode=eval_mode,
                        attempt_id=attempt_id,
                        run_dir=run_dir,
                        run_slug=run_slug,
                        llm_config=llm_config,
                        op_summary_attempts=dict(op_summary.get("attempts") or {}),
                        curr_outcome=outcome.__dict__,
                        curr_payload=payload,
                        prev_code=saved_prev_for_memory,
                        curr_code=gen["result"].generated_code or "",
                        memory_root=memory_root,
                        max_log_lines=max_log_lines,
                    )
                except Exception:
                    pass

            previous_attempt_code = gen["result"].generated_code
            previous_repair_context_path = str(repair_path)

            attempt_failed = _attempt_failed(outcome.compiled, outcome.correctness)
            if not attempt_failed:
                op_summary["fixed_on_attempt"] = attempt_id
                if attempt_id == 1:
                    op_summary["final_status"] = "pass_on_attempt1"
                else:
                    op_summary["final_status"] = f"fixed_on_attempt{attempt_id}"
                    op_summary["fixed_in_attempt2"] = attempt_id == 2
        except Exception:
            outcome.generation_error = traceback.format_exc()
            if attempt_id == 1:
                op_summary["final_status"] = "generation_failed_on_attempt1"
            else:
                op_summary["final_status"] = f"attempt{attempt_id}_generation_or_eval_failed"
            op_summary["attempts"][f"attempt{attempt_id}"] = outcome.__dict__
            break

        op_summary["attempts"][f"attempt{attempt_id}"] = outcome.__dict__
        if op_summary["fixed_on_attempt"] is not None:
            break

        if attempt_id == max_attempts:
            op_summary["final_status"] = f"failed_after_attempt{max_attempts}"

    if op_summary["final_status"] == "unknown":
        op_summary["final_status"] = f"failed_after_attempt{max_attempts}"
    return op_summary


def _run_single_cuda_agent_job(
    *,
    op_key: str,
    row_index: int,
    row: Dict[str, Any],
    dataset_path: pathlib.Path,
    strategy: str,
    tool_mode: str,
    llm_config: Dict[str, Any],
    run_dir: pathlib.Path,
    eval_mode: str,
    clean_policy: str,
    max_log_lines: int,
    max_attempts: int,
    eval_workers: int,
    eval_npu: int,
    op_slot: int,
    parallel_ops: int,
    ascend_search_version_filter: Optional[str] = None,
    run_slug: str = "",
    memory_root: Optional[pathlib.Path] = None,
    use_repair_memory: bool = False,
    start_attempt_id: int = 1,
    seed_previous_code: str = "",
    seed_repair_context_path: str = "",
    prior_attempts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from tools.common.env import apply_agent_parallel_slot_env

    apply_agent_parallel_slot_env(op_slot=op_slot, parallel_ops=parallel_ops, npu_count=eval_npu)
    print(
        f"[ALLOC] op_key={op_key} slot={op_slot} ASCEND_VISIBLE_DEVICES="
        f"{os.environ.get('ASCEND_VISIBLE_DEVICES')} (eval_npu={eval_npu})"
    )
    category = str(row.get("data_source") or "cuda_agent")
    return run_multi_attempt_for_cuda_row(
        op_key=op_key,
        row_index=row_index,
        row=row,
        dataset_path=dataset_path,
        category=category,
        strategy=strategy,
        tool_mode=tool_mode,
        llm_config=llm_config,
        run_dir=run_dir,
        eval_mode=eval_mode,
        clean_policy=clean_policy,
        max_log_lines=max_log_lines,
        max_attempts=max_attempts,
        eval_workers=eval_workers,
        eval_npu=eval_npu,
        ascend_search_version_filter=ascend_search_version_filter,
        run_slug=run_slug,
        memory_root=memory_root,
        use_repair_memory=use_repair_memory,
        start_attempt_id=start_attempt_id,
        seed_previous_code=seed_previous_code,
        seed_repair_context_path=seed_repair_context_path,
        prior_attempts=prior_attempts,
    )


def _default_cuda_agent_run_dir(
    *,
    model_slug: str,
    tool_mode: str,
    strategy: str,
    run: int,
    test: bool,
    use_repair_memory: bool = False,
) -> pathlib.Path:
    from generator.agent.agent_config import parse_tool_mode, tool_mode_to_string

    if use_repair_memory:
        out_root = pathlib.Path(
            "output/test/memory_on/cuda_agent_ops_6k" if test else "output/memory_on/cuda_agent_ops_6k"
        )
    else:
        out_root = pathlib.Path("output/test/cuda_agent_ops_6k" if test else "output/cuda_agent_ops_6k")
    return (
        out_root
        / model_slug
        / f"agent_{tool_mode_to_string(parse_tool_mode(tool_mode))}"
        / strategy
        / f"run{run}"
    )


def _per_op_summary_filename(op: str, max_attempts: int) -> str:
    return f"{op}_attempts{max_attempts}_summary.json"


def _all_ops_summary_filename(max_attempts: int) -> str:
    return f"attempts{max_attempts}_summary_all_rows.json"


def _continue_single_cuda_worker(
    ent_dict: Dict[str, Any],
    *,
    args_dict: Dict[str, Any],
    out_run_dir_str: str,
    dataset_path_str: str,
    llm_config: Dict[str, Any],
    ascend_vf: Optional[str],
    memory_run_slug: str,
    memory_root_str: str,
    continued_from_abs: str,
    continue_session_at: str,
    new_max_attempts: int,
    op_slot: int,
) -> Tuple[str, Dict[str, Any]]:
    from generator.scripts.multi_round_continue import merge_op_summary
    from tools.cuda_agent_eval.dataset_snapshot import load_dataset_row

    out_run_dir = pathlib.Path(out_run_dir_str)
    dataset_path = pathlib.Path(dataset_path_str)
    memory_root = pathlib.Path(memory_root_str) if memory_root_str else None
    row_index = int(ent_dict["row_index"])
    row = load_dataset_row(dataset_path, row_index)
    seed_code = ""
    if ent_dict.get("seed_txt_path"):
        seed_code = pathlib.Path(ent_dict["seed_txt_path"]).read_text(encoding="utf-8", errors="replace")
    new_partial = _run_single_cuda_agent_job(
        op_key=ent_dict["entity_key"],
        row_index=row_index,
        row=row,
        dataset_path=dataset_path,
        strategy=args_dict["strategy"],
        tool_mode=args_dict["tool_mode"],
        llm_config=llm_config,
        run_dir=out_run_dir,
        eval_mode=args_dict["eval_mode"],
        clean_policy=args_dict["clean_policy"],
        max_log_lines=args_dict["max_log_lines"],
        max_attempts=new_max_attempts,
        eval_workers=args_dict["eval_workers"],
        eval_npu=args_dict["eval_npu"],
        op_slot=op_slot,
        parallel_ops=args_dict["parallel_ops"],
        ascend_search_version_filter=ascend_vf,
        run_slug=memory_run_slug,
        memory_root=memory_root,
        use_repair_memory=args_dict["use_repair_memory"],
        start_attempt_id=int(ent_dict["last_attempt_id"]) + 1,
        seed_previous_code=seed_code,
        seed_repair_context_path=str(ent_dict.get("seed_repair_context_path") or ""),
        prior_attempts=ent_dict.get("prior_attempts") or {},
    )
    merged = merge_op_summary(
        ent_dict.get("prior_op_summary") or {},
        new_partial,
        new_max_attempts=new_max_attempts,
        continued_from_abs=continued_from_abs,
        source_last_attempt=int(ent_dict["last_attempt_id"]),
        continue_session_at=continue_session_at,
        ran_new_attempts=True,
    )
    return ent_dict["entity_key"], merged


def _run_continue_session_cuda(
    *,
    args: argparse.Namespace,
    out_run_dir: pathlib.Path,
    dataset_path: pathlib.Path,
    llm_config: Dict[str, Any],
    resolved_model: str,
    ascend_vf: Optional[str],
    memory_run_slug: str,
    memory_root: Optional[pathlib.Path],
) -> int:
    from generator.scripts.multi_round_continue import (
        OpContinueState,
        aggregate_summary_filename,
        build_continue_plan,
        merge_op_summary,
        per_entity_summary_filename,
        utc_now_iso,
        write_continue_report,
    )
    from tools.cuda_agent_eval.dataset_snapshot import load_dataset_row

    continue_session_at = utc_now_iso()
    row_keys_filter: Optional[List[str]] = None
    if getattr(args, "row_keys", None):
        row_keys_filter = list(args.row_keys)
    elif args.indices is not None:
        from tools.cuda_agent_eval.constants import suggested_op_key_ca6k

        row_keys_filter = [suggested_op_key_ca6k(int(i)) for i in args.indices]

    try:
        plan = build_continue_plan(
            run_dir=out_run_dir,
            new_max_attempts=args.max_attempts,
            kind="cuda",
            ops_filter=row_keys_filter,
            max_log_lines=args.max_log_lines,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 2

    to_run = [e for e in plan.entities if e.action == "continue"]
    if not to_run:
        print("[ERROR] No rows to continue.")
        return 2

    entity_results: Dict[str, Dict[str, Any]] = {}
    merged_summaries: Dict[str, Dict[str, Any]] = {}
    order_index = {e.entity_key: i for i, e in enumerate(plan.entities)}

    def _process_entity(ent: OpContinueState, op_slot: int) -> Dict[str, Any]:
        if ent.action != "continue":
            return merge_op_summary(
                ent.prior_op_summary,
                {},
                new_max_attempts=args.max_attempts,
                continued_from_abs=plan.continued_from_abs,
                source_last_attempt=ent.last_attempt_id,
                continue_session_at=continue_session_at,
                ran_new_attempts=False,
            )
        if ent.row_index is None:
            raise RuntimeError(f"row_index missing for {ent.entity_key}")
        row = load_dataset_row(dataset_path, ent.row_index)
        seed_code = ent.seed_txt_path.read_text(encoding="utf-8", errors="replace") if ent.seed_txt_path else ""
        print(
            f"[RUN] op_key={ent.entity_key} row_index={ent.row_index} "
            f"continue attempt{ent.last_attempt_id + 1}..{args.max_attempts}"
        )
        new_partial = _run_single_cuda_agent_job(
            op_key=ent.entity_key,
            row_index=ent.row_index,
            row=row,
            dataset_path=dataset_path,
            strategy=args.strategy,
            tool_mode=args.tool_mode,
            llm_config=llm_config,
            run_dir=out_run_dir,
            eval_mode=args.eval_mode,
            clean_policy=args.clean_policy,
            max_log_lines=args.max_log_lines,
            max_attempts=args.max_attempts,
            eval_workers=args.eval_workers,
            eval_npu=args.eval_npu,
            op_slot=op_slot,
            parallel_ops=args.parallel_ops,
            ascend_search_version_filter=ascend_vf,
            run_slug=memory_run_slug,
            memory_root=memory_root,
            use_repair_memory=args.use_repair_memory,
            start_attempt_id=ent.last_attempt_id + 1,
            seed_previous_code=seed_code,
            seed_repair_context_path=str(ent.seed_repair_context_path or ""),
            prior_attempts=ent.prior_attempts,
        )
        return merge_op_summary(
            ent.prior_op_summary,
            new_partial,
            new_max_attempts=args.max_attempts,
            continued_from_abs=plan.continued_from_abs,
            source_last_attempt=ent.last_attempt_id,
            continue_session_at=continue_session_at,
            ran_new_attempts=True,
        )

    if args.parallel_ops == 1:
        slot = 0
        for ent in plan.entities:
            merged = _process_entity(ent, slot)
            merged_summaries[ent.entity_key] = merged
            entity_results[ent.entity_key] = merged
            if ent.action == "continue":
                slot += 1
    else:
        args_dict = {
            "strategy": args.strategy,
            "tool_mode": args.tool_mode,
            "eval_mode": args.eval_mode,
            "clean_policy": args.clean_policy,
            "max_log_lines": args.max_log_lines,
            "eval_workers": args.eval_workers,
            "eval_npu": args.eval_npu,
            "parallel_ops": args.parallel_ops,
            "use_repair_memory": args.use_repair_memory,
        }
        memory_root_str = str(memory_root) if memory_root else ""
        continue_entities = [e for e in plan.entities if e.action == "continue"]
        slot_map = {e.entity_key: i for i, e in enumerate(continue_entities)}
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_ops) as ex:
            fut_to_key = {}
            for ent in plan.entities:
                if ent.action != "continue":
                    merged_summaries[ent.entity_key] = _process_entity(ent, 0)
                    entity_results[ent.entity_key] = merged_summaries[ent.entity_key]
                    continue
                ent_dict = {
                    "entity_key": ent.entity_key,
                    "row_index": ent.row_index,
                    "last_attempt_id": ent.last_attempt_id,
                    "seed_txt_path": str(ent.seed_txt_path) if ent.seed_txt_path else "",
                    "seed_repair_context_path": str(ent.seed_repair_context_path or ""),
                    "prior_attempts": ent.prior_attempts,
                    "prior_op_summary": ent.prior_op_summary,
                }
                fut = ex.submit(
                    _continue_single_cuda_worker,
                    ent_dict,
                    args_dict=args_dict,
                    out_run_dir_str=str(out_run_dir),
                    dataset_path_str=str(dataset_path),
                    llm_config=llm_config,
                    ascend_vf=ascend_vf,
                    memory_run_slug=memory_run_slug,
                    memory_root_str=memory_root_str,
                    continued_from_abs=plan.continued_from_abs,
                    continue_session_at=continue_session_at,
                    new_max_attempts=args.max_attempts,
                    op_slot=slot_map[ent.entity_key],
                )
                fut_to_key[fut] = ent.entity_key
            for fut in concurrent.futures.as_completed(fut_to_key):
                key = fut_to_key[fut]
                try:
                    _, merged = fut.result()
                    merged_summaries[key] = merged
                except Exception:
                    ent = next(e for e in plan.entities if e.entity_key == key)
                    merged_summaries[key] = merge_op_summary(
                        ent.prior_op_summary,
                        {
                            "op_key": key,
                            "final_status": "op_level_parallel_runtime_failed",
                            "error": traceback.format_exc(),
                        },
                        new_max_attempts=args.max_attempts,
                        continued_from_abs=plan.continued_from_abs,
                        source_last_attempt=ent.last_attempt_id,
                        continue_session_at=continue_session_at,
                        ran_new_attempts=True,
                    )
                entity_results[key] = merged_summaries[key]

    prior_agg = plan.previous_aggregate
    resolved_indices = []
    for ent in plan.entities:
        ri = merged_summaries[ent.entity_key].get("row_index") or ent.row_index
        if ri is not None:
            resolved_indices.append(int(ri))

    all_summaries: Dict[str, Any] = {
        "kind": "cuda_agent_ops_6k_multi_round",
        "model": resolved_model,
        "dataset_path": str(dataset_path),
        "tool_mode": args.tool_mode,
        "ascend_search_version_filter": ascend_vf,
        "strategy": args.strategy,
        "eval_mode": args.eval_mode,
        "clean_policy": args.clean_policy,
        "max_attempts": args.max_attempts,
        "parallel_ops": args.parallel_ops,
        "eval_workers": args.eval_workers,
        "eval_npu": args.eval_npu,
        "run_dir": str(out_run_dir),
        "resolved_indices": resolved_indices,
        "op_counts_filter": prior_agg.get("op_counts_filter"),
        "rows": {},
        "use_repair_memory": args.use_repair_memory,
    }

    report_path = write_continue_report(
        run_dir=out_run_dir,
        plan=plan,
        continue_session_at=continue_session_at,
        model=resolved_model,
        use_repair_memory=args.use_repair_memory,
        entity_results=entity_results,
    )
    all_summaries["continue"] = {
        "enabled": True,
        "continued_from": plan.continued_from_abs,
        "previous_max_attempts": plan.previous_max_attempts,
        "new_max_attempts": args.max_attempts,
        "continue_report": report_path.name,
    }

    for ent in sorted(plan.entities, key=lambda e: order_index.get(e.entity_key, 10**9)):
        summary = merged_summaries[ent.entity_key]
        all_summaries["rows"][ent.entity_key] = summary
        per_path = out_run_dir / per_entity_summary_filename(ent.entity_key, args.max_attempts)
        _write_json(per_path, summary)
        print(f"[WROTE] {per_path}")

    agg = out_run_dir / aggregate_summary_filename("cuda", args.max_attempts)
    _write_json(agg, all_summaries)
    print(f"[WROTE] {agg}")
    print(f"[WROTE] {report_path}")

    if args.use_repair_memory:
        try:
            from generator.repair_memory import merge_run_inbox
            from generator.repair_memory.paths import get_memory_root

            merged_n = merge_run_inbox(get_memory_root(), memory_run_slug)
            if merged_n:
                print(f"[REPAIR_MEMORY] final merge: appended {merged_n} record(s) (run_slug={memory_run_slug})")
        except Exception as exc:
            print(f"[REPAIR_MEMORY] final merge skipped: {exc}")
    return 0


def main() -> int:
    from generator.agent.agent_config import get_llm_config_compatible, model_slug_for_path
    from generator.cuda_agent_dataset_selection import resolve_row_indices
    from tools.cuda_agent_eval.constants import suggested_op_key_ca6k
    from tools.cuda_agent_eval.dataset_snapshot import load_dataset_row

    parser = argparse.ArgumentParser(
        description="CUDA-Agent-Ops-6K: multi-attempt agent generation + eval_cuda_agent_operator + repair.",
        epilog=(
            "Requires exactly one of --indices, --range, or --all. "
            "Inherits environment for API config (USE_API_CONFIG), reference device (LLM4ASCENDC_REF_ON_CPU), "
            "artifacts (LLM4ASCENDC_CUDA_AGENT_ART_ROOT), OPP path (LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH)."
        ),
    )
    sel = parser.add_mutually_exclusive_group(required=False)
    sel.add_argument(
        "--indices",
        nargs="+",
        type=int,
        metavar="N",
        help="0-based jsonl row indices (same as eval_cuda_agent_operator --row-index).",
    )
    sel.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive row index range [START, END].",
    )
    sel.add_argument(
        "--all",
        action="store_true",
        help="Process every row in the jsonl (use with care).",
    )
    parser.add_argument(
        "--op-counts",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="Keep only rows whose ops list has length in this set (e.g. 1 2). Omit for no filter.",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=REPO_ROOT / "data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl",
        help="CUDA-Agent-Ops-6K jsonl (default: bundled path under data/external/).",
    )
    parser.add_argument("--tool-mode", type=str, default="all", help="Tool mode string.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["one_shot", "none"],
        default="one_shot",
        help=(
            "CUDA-Agent fused prompt only: one_shot inserts leaky_relu few-shot (same as single-op ascendc one_shot); "
            "none skips the example block. Output path includes this segment (…/one_shot/runN vs …/none/runN)."
        ),
    )
    parser.add_argument("--run", type=int, default=0, help="Run index for output path.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Custom run directory (contains attempt1..attemptN). Overrides default under output/cuda_agent_ops_6k/.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use output/test/cuda_agent_ops_6k/ when --out-dir is not set.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override LLM model.")
    parser.add_argument("--eval-mode", choices=["full", "build-only", "eval-only"], default="full")
    parser.add_argument("--clean-policy", choices=["force", "smart"], default="force")
    parser.add_argument("--max-log-lines", type=int, default=220, help="Tail lines per selected log in repair context.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum generation/eval attempts per row (>=2).")
    parser.add_argument("--parallel-ops", type=int, default=1, help="Number of rows to run in parallel.")
    parser.add_argument("--eval-workers", type=int, default=1, help="Pass-through eval workers per attempt.")
    parser.add_argument("--eval-npu", type=int, default=1, help="Pass-through eval NPU count per attempt.")
    parser.add_argument(
        "--ascend-search-version",
        default=None,
        metavar="SUBSTRING",
        help="Ascend docs version substring filter for ascend_search + ascend_fetch.",
    )
    parser.add_argument(
        "--use-repair-memory",
        action="store_true",
        help=(
            "Enable cross-attempt repair memory; default output under output/memory_on/cuda_agent_ops_6k/... "
            "(with --test: output/test/memory_on/cuda_agent_ops_6k/...). Omit for legacy behavior without memory."
        ),
    )
    parser.add_argument(
        "--continue-attempt",
        action="store_true",
        help="Continue multi-round repair in an existing run directory (append attemptN+1..max).",
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default="",
        help="Existing run directory to continue from (required with --continue-attempt).",
    )
    parser.add_argument(
        "--row-keys",
        nargs="+",
        default=None,
        metavar="KEY",
        help="In continue mode: subset of ca6k row keys (e.g. ca6k_00055).",
    )
    args = parser.parse_args()
    if args.continue_attempt and not (args.continue_from or "").strip():
        raise ValueError("--continue-from is required when --continue-attempt is set")
    if (args.continue_from or "").strip() and not args.continue_attempt:
        raise ValueError("--continue-attempt is required when --continue-from is set")

    if args.max_attempts < 2:
        raise ValueError("--max-attempts must be >= 2")
    if args.parallel_ops < 1:
        raise ValueError("--parallel-ops must be >= 1")
    if args.eval_workers < 1:
        raise ValueError("--eval-workers must be >= 1")
    if args.eval_npu < 1:
        raise ValueError("--eval-npu must be >= 1")
    if args.eval_workers != 1:
        raise ValueError("--eval-workers currently only supports value 1 for per-attempt eval.")
    if args.parallel_ops > args.eval_npu:
        print(
            f"[WARN] parallel_ops ({args.parallel_ops}) > eval_npu ({args.eval_npu}); "
            "multiple jobs will share visible devices."
        )
    if args.parallel_ops > 1:
        from tools.common.env import ensure_parallel_build_jobs

        jobs = ensure_parallel_build_jobs(worker_count=args.parallel_ops)
        print(f"[batch] LLM4ASCENDC_BUILD_JOBS={jobs} (parallel_ops={args.parallel_ops})")

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    llm_config = get_llm_config_compatible(cli_model=args.model)
    resolved_model = llm_config["model"]
    model_slug = model_slug_for_path(resolved_model)
    ascend_vf = _normalize_ascend_search_version(args.ascend_search_version)
    if ascend_vf is not None:
        from generator.agent.agent_config import has_ascend_fetch, has_ascend_search, parse_tool_mode

        _tm = parse_tool_mode(args.tool_mode)
        if not (has_ascend_search(_tm) and has_ascend_fetch(_tm)):
            print(
                "[WARN] --ascend-search-version works best with ascend_search + ascend_fetch; "
                f"ascend_search={has_ascend_search(_tm)}, ascend_fetch={has_ascend_fetch(_tm)}."
            )

    if args.continue_attempt:
        out_run_dir = pathlib.Path(args.continue_from).resolve()
        if not out_run_dir.is_dir():
            raise ValueError(f"--continue-from is not a directory: {out_run_dir}")
        if args.out_dir:
            out_explicit = pathlib.Path(args.out_dir).resolve()
            if out_explicit != out_run_dir:
                raise ValueError(
                    f"--out-dir ({out_explicit}) must match --continue-from ({out_run_dir}) in continue mode"
                )
        print(
            f"[INFO] Continue mode: run_dir={out_run_dir} "
            "(ignoring --run, --range, --all; optional --indices/--row-keys filter)"
        )
    else:
        range_pair: Optional[tuple[int, int]] = None
        use_all = False
        indices_arg: Optional[list[int]] = None
        if args.indices is not None:
            indices_arg = list(args.indices)
        elif args.range is not None:
            range_pair = (int(args.range[0]), int(args.range[1]))
        elif args.all:
            use_all = True
        else:
            raise ValueError("Exactly one of --indices, --range, or --all is required (unless --continue-attempt)")

        try:
            resolved_indices = resolve_row_indices(
                dataset_path,
                indices=indices_arg,
                range_pair=range_pair,
                use_all=use_all,
                op_counts=list(args.op_counts) if args.op_counts is not None else None,
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            raise SystemExit(2) from e

        if not resolved_indices:
            print("[ERROR] No rows to run after filters (--indices/--range/--all and optional --op-counts).")
            raise SystemExit(2)

        out_run_dir = (
            pathlib.Path(args.out_dir)
            if args.out_dir
            else _default_cuda_agent_run_dir(
                model_slug=model_slug,
                tool_mode=args.tool_mode,
                strategy=args.strategy,
                run=args.run,
                test=args.test,
                use_repair_memory=args.use_repair_memory,
            )
        )
        out_run_dir.mkdir(parents=True, exist_ok=True)
    if args.use_repair_memory and args.out_dir and not args.continue_attempt:
        p = str(out_run_dir.resolve()).replace("\\", "/")
        if "memory_on" not in p:
            print(
                "[WARN] With --use-repair-memory, prefer run_dir under a path containing memory_on "
                "(default when --out-dir is unset); "
                f"current run_dir={out_run_dir}"
            )

    from generator.repair_memory.paths import get_memory_root, run_slug_from_run_dir

    memory_run_slug = run_slug_from_run_dir(out_run_dir) if args.use_repair_memory else ""
    memory_root: Optional[pathlib.Path] = get_memory_root() if args.use_repair_memory else None

    if args.continue_attempt:
        return _run_continue_session_cuda(
            args=args,
            out_run_dir=out_run_dir,
            dataset_path=dataset_path,
            llm_config=llm_config,
            resolved_model=resolved_model,
            ascend_vf=ascend_vf,
            memory_run_slug=memory_run_slug,
            memory_root=memory_root,
        )

    print(f"[INFO] Rows to run ({len(resolved_indices)}); dataset={dataset_path}")
    if len(resolved_indices) <= 40:
        print(f"[INFO] row indices: {resolved_indices}")

    rows_payload: Dict[int, Dict[str, Any]] = {}
    for idx in resolved_indices:
        rows_payload[idx] = load_dataset_row(dataset_path, idx)

    op_keys = [suggested_op_key_ca6k(i) for i in resolved_indices]

    all_summaries: Dict[str, Any] = {
        "kind": "cuda_agent_ops_6k_multi_round",
        "model": resolved_model,
        "dataset_path": str(dataset_path),
        "tool_mode": args.tool_mode,
        "ascend_search_version_filter": ascend_vf,
        "strategy": args.strategy,
        "eval_mode": args.eval_mode,
        "clean_policy": args.clean_policy,
        "max_attempts": args.max_attempts,
        "parallel_ops": args.parallel_ops,
        "eval_workers": args.eval_workers,
        "eval_npu": args.eval_npu,
        "run_dir": str(out_run_dir),
        "resolved_indices": resolved_indices,
        "op_counts_filter": list(args.op_counts) if args.op_counts is not None else None,
        "rows": {},
        "use_repair_memory": args.use_repair_memory,
    }

    if args.parallel_ops == 1:
        results = []
        for op_slot, (row_index, op_key) in enumerate(zip(resolved_indices, op_keys)):
            row = rows_payload[row_index]
            print(f"[RUN] row_index={row_index} op_key={op_key} strategy={args.strategy} tool_mode={args.tool_mode}")
            summary = _run_single_cuda_agent_job(
                op_key=op_key,
                row_index=row_index,
                row=row,
                dataset_path=dataset_path,
                strategy=args.strategy,
                tool_mode=args.tool_mode,
                llm_config=llm_config,
                run_dir=out_run_dir,
                eval_mode=args.eval_mode,
                clean_policy=args.clean_policy,
                max_log_lines=args.max_log_lines,
                max_attempts=args.max_attempts,
                eval_workers=args.eval_workers,
                eval_npu=args.eval_npu,
                op_slot=op_slot,
                parallel_ops=args.parallel_ops,
                ascend_search_version_filter=ascend_vf,
                run_slug=memory_run_slug,
                use_repair_memory=args.use_repair_memory,
            )
            results.append({"op_key": op_key, "row_index": row_index, "summary": summary})
    else:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_ops) as ex:
            fut_to_key: Dict[concurrent.futures.Future[Any], tuple[str, int]] = {}
            for op_slot, (row_index, op_key) in enumerate(zip(resolved_indices, op_keys)):
                row = rows_payload[row_index]
                print(
                    f"[RUN] row_index={row_index} op_key={op_key} strategy={args.strategy} tool_mode={args.tool_mode}"
                )
                fut = ex.submit(
                    _run_single_cuda_agent_job,
                    op_key=op_key,
                    row_index=row_index,
                    row=row,
                    dataset_path=dataset_path,
                    strategy=args.strategy,
                    tool_mode=args.tool_mode,
                    llm_config=llm_config,
                    run_dir=out_run_dir,
                    eval_mode=args.eval_mode,
                    clean_policy=args.clean_policy,
                    max_log_lines=args.max_log_lines,
                    max_attempts=args.max_attempts,
                    eval_workers=args.eval_workers,
                    eval_npu=args.eval_npu,
                    op_slot=op_slot,
                    parallel_ops=args.parallel_ops,
                    ascend_search_version_filter=ascend_vf,
                    run_slug=memory_run_slug,
                    use_repair_memory=args.use_repair_memory,
                )
                fut_to_key[fut] = (op_key, row_index)
            for fut in concurrent.futures.as_completed(fut_to_key):
                op_key, row_index = fut_to_key[fut]
                try:
                    summary = fut.result()
                    results.append({"op_key": op_key, "row_index": row_index, "summary": summary})
                except Exception:
                    err_summary = {
                        "op_key": op_key,
                        "row_index": row_index,
                        "category": "cuda_agent",
                        "attempts": {},
                        "fixed_in_attempt2": False,
                        "fixed_on_attempt": None,
                        "max_attempts": args.max_attempts,
                        "final_status": "op_level_parallel_runtime_failed",
                        "error": traceback.format_exc(),
                    }
                    results.append({"op_key": op_key, "row_index": row_index, "summary": err_summary})

    order_index = {op_key: i for i, op_key in enumerate(op_keys)}
    for item in sorted(results, key=lambda x: order_index.get(x["op_key"], 10**9)):
        op_key = item["op_key"]
        summary = item["summary"]
        all_summaries["rows"][op_key] = summary
        per_path = out_run_dir / _per_op_summary_filename(op_key, args.max_attempts)
        _write_json(per_path, summary)
        print(f"[WROTE] {per_path}")

    agg = out_run_dir / _all_ops_summary_filename(args.max_attempts)
    _write_json(agg, all_summaries)
    print(f"[WROTE] {agg}")
    if args.use_repair_memory:
        try:
            from generator.repair_memory import merge_run_inbox
            from generator.repair_memory.paths import get_memory_root

            merged_n = merge_run_inbox(get_memory_root(), memory_run_slug)
            if merged_n:
                print(f"[REPAIR_MEMORY] final merge: appended {merged_n} record(s) (run_slug={memory_run_slug})")
        except Exception as exc:
            print(f"[REPAIR_MEMORY] final merge skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
