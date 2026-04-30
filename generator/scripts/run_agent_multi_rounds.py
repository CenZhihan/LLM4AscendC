#!/usr/bin/env python3
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
from typing import Dict, Any, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


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


def _eval_result_json_path(txt_path: pathlib.Path, op: str) -> pathlib.Path:
    rel_group = _artifact_group_rel_from_txt_path(txt_path)
    art_root = ARTIFACTS_ROOT / rel_group if rel_group is not None else ARTIFACTS_ROOT
    return art_root / op / f"result_{op}.json"


def _run_eval_for_txt(
    txt_path: pathlib.Path,
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
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "eval_operator.py")]
    cmd.extend(
        [
            "--txt",
            str(txt_path),
            "--mode",
            mode,
            "--clean-policy",
            clean_policy,
        ]
    )
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


def _load_result_payload(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_eval_core(payload: Dict[str, Any], op: str) -> Dict[str, Any]:
    result = (payload.get("result") or {}).get(op) or {}
    meta = payload.get("meta") or {}
    return {
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "correctness_info": result.get("correctness_info") or "",
        "logs": meta.get("logs") or {},
    }


def _select_error_logs(logs: Dict[str, str]) -> List[str]:
    preferred = ["02-build", "06-eval"]
    selected: List[str] = []
    for key in preferred:
        value = (logs.get(key) or "").strip()
        if value:
            selected.append(value)
    if selected:
        return selected

    fallback_order = ["01-msopgen", "03-install-run", "04-pybind-build", "05-pybind-install"]
    for key in fallback_order:
        value = (logs.get(key) or "").strip()
        if value:
            selected.append(value)
    return selected


def _tail_lines(raw: str, max_lines: int) -> str:
    if max_lines <= 0:
        return raw
    lines = raw.splitlines()
    if len(lines) <= max_lines:
        return raw
    return "\n".join(lines[-max_lines:])


def _build_repair_error_context(
    *,
    op: str,
    result_payload: Dict[str, Any],
    selected_logs: List[str],
    max_log_lines: int,
) -> str:
    core = _extract_eval_core(result_payload, op)
    sections: List[str] = []
    sections.append(f"=== attempt1 eval summary for {op} ===")
    sections.append(f"compiled: {core['compiled']}")
    sections.append(f"correctness: {core['correctness']}")
    sections.append("")
    if core["correctness_info"]:
        sections.append("=== correctness_info (raw text) ===")
        sections.append(core["correctness_info"])
        sections.append("")
    for p in selected_logs:
        path = pathlib.Path(p)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            sections.append(f"=== log read failed: {p} ({exc}) ===")
            sections.append("")
            continue
        trimmed = _tail_lines(raw, max_lines=max_log_lines)
        sections.append(f"=== log file: {p} (tail {max_log_lines} lines) ===")
        sections.append(trimmed)
        sections.append("")
    return "\n".join(sections).strip() + "\n"


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


def _generate_one_attempt(
    *,
    op: str,
    category: str,
    strategy: str,
    tool_mode: str,
    llm_config: Dict[str, Any],
    out_dir: pathlib.Path,
    attempt_id: int,
    repair_error_logs_raw: str = "",
    previous_attempt_code: str = "",
) -> Dict[str, Any]:
    from generator.agent.agent_runner import KernelGenerationTask, generate_kernel_with_agent

    task = KernelGenerationTask(
        language="ascendc",
        op=op,
        strategy_name=strategy,
        category=category,
    )
    result = generate_kernel_with_agent(
        task,
        tool_mode=tool_mode,
        llm_config=llm_config,
        attempt_id=attempt_id,
        repair_error_logs_raw=repair_error_logs_raw,
        previous_attempt_code=previous_attempt_code,
    )
    paths = _save_generation_outputs(
        out_dir=out_dir,
        op=op,
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


def run_multi_attempt_for_op(
    *,
    op: str,
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
) -> Dict[str, Any]:
    if max_attempts < 2:
        raise ValueError("max_attempts must be >= 2")
    op_summary: Dict[str, Any] = {
        "op": op,
        "category": category,
        "attempts": {},
        "fixed_in_attempt2": False,
        "fixed_on_attempt": None,
        "max_attempts": max_attempts,
        "final_status": "unknown",
    }
    previous_attempt_code = ""
    previous_repair_context_path = ""

    for attempt_id in range(1, max_attempts + 1):
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
            repair_logs_raw = ""
            if attempt_id >= 2:
                if not previous_repair_context_path:
                    raise RuntimeError(f"attempt{attempt_id} missing previous repair context")
                repair_logs_raw = pathlib.Path(previous_repair_context_path).read_text(
                    encoding="utf-8", errors="replace"
                )
            gen = _generate_one_attempt(
                op=op,
                category=category,
                strategy=strategy,
                tool_mode=tool_mode,
                llm_config=llm_config,
                out_dir=attempt_dir,
                attempt_id=attempt_id,
                repair_error_logs_raw=repair_logs_raw,
                previous_attempt_code=previous_attempt_code,
            )
            outcome.generated = True
            outcome.txt_path = str(gen["txt_path"])
            outcome.report_path = str(gen["report_path"])

            eval_rc = _run_eval_for_txt(
                gen["txt_path"],
                mode=eval_mode,
                clean_policy=clean_policy,
                eval_workers=eval_workers,
                eval_npu=eval_npu,
            )
            outcome.eval_ran = True
            result_path = _eval_result_json_path(gen["txt_path"], op)
            if not result_path.exists():
                raise RuntimeError(
                    f"eval finished (rc={eval_rc}) but result json not found: {result_path}"
                )
            outcome.eval_result_path = str(result_path)
            payload = _load_result_payload(result_path)
            core = _extract_eval_core(payload, op)
            outcome.compiled = core["compiled"]
            outcome.correctness = core["correctness"]
            outcome.correctness_info = core["correctness_info"]
            outcome.selected_log_paths = _select_error_logs(core["logs"])

            repair_text = _build_repair_error_context(
                op=op,
                result_payload=payload,
                selected_logs=outcome.selected_log_paths,
                max_log_lines=max_log_lines,
            )
            repair_path = attempt_dir / f"{op}_repair_context.txt"
            _write_text(repair_path, repair_text)
            outcome.repair_context_path = str(repair_path)

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


def _run_single_op_job(
    *,
    op: str,
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
    op_slot: int,
) -> Dict[str, Any]:
    device_id = op_slot % eval_npu
    os.environ["ASCEND_VISIBLE_DEVICES"] = str(device_id)
    print(f"[ALLOC] op={op} slot={op_slot} ASCEND_VISIBLE_DEVICES={device_id} (eval_npu={eval_npu})")
    return run_multi_attempt_for_op(
        op=op,
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
    )


def _default_run_dir(*, model_slug: str, tool_mode: str, strategy: str, run: int, test: bool) -> pathlib.Path:
    from generator.agent.agent_config import parse_tool_mode, tool_mode_to_string

    out_root = pathlib.Path("output/test/ascendc" if test else "output/ascendc")
    return out_root / model_slug / f"agent_{tool_mode_to_string(parse_tool_mode(tool_mode))}" / strategy / f"run{run}"


def _per_op_summary_filename(op: str, max_attempts: int) -> str:
    return f"{op}_attempts{max_attempts}_summary.json"


def _all_ops_summary_filename(max_attempts: int) -> str:
    return f"attempts{max_attempts}_summary_all_ops.json"


def main() -> int:
    from vendor.mkb.dataset import dataset
    from generator.agent.agent_config import get_llm_config_compatible, model_slug_for_path

    parser = argparse.ArgumentParser(description="Automated multi-attempt agent generation + eval + repair.")
    parser.add_argument("--ops", nargs="+", required=True, help="Operator keys to run.")
    parser.add_argument("--tool-mode", type=str, default="all", help="Tool mode string.")
    parser.add_argument("--strategy", type=str, default="one_shot", help="Prompt strategy.")
    parser.add_argument("--run", type=int, default=0, help="Run index for output path.")
    parser.add_argument("--out-dir", type=str, default="", help="Custom run directory (contains attempt1..attemptN).")
    parser.add_argument("--test", action="store_true", help="Use output/test/ascendc when --out-dir is not set.")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model.")
    parser.add_argument("--eval-mode", choices=["full", "build-only", "eval-only"], default="full")
    parser.add_argument("--clean-policy", choices=["force", "smart"], default="force")
    parser.add_argument("--max-log-lines", type=int, default=220, help="Tail lines per selected log in repair context.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum generation/eval attempts per op (>=2).")
    parser.add_argument("--parallel-ops", type=int, default=1, help="Number of ops to run in parallel.")
    parser.add_argument("--eval-workers", type=int, default=1, help="Pass-through eval workers per attempt.")
    parser.add_argument("--eval-npu", type=int, default=1, help="Pass-through eval npu count per attempt.")
    args = parser.parse_args()
    if args.max_attempts < 2:
        raise ValueError("--max-attempts must be >= 2")
    if args.parallel_ops < 1:
        raise ValueError("--parallel-ops must be >= 1")
    if args.eval_workers < 1:
        raise ValueError("--eval-workers must be >= 1")
    if args.eval_npu < 1:
        raise ValueError("--eval-npu must be >= 1")
    if args.eval_workers != 1:
        raise ValueError("--eval-workers currently only supports value 1 for per-op attempt eval.")
    if args.parallel_ops > args.eval_npu:
        print(
            f"[WARN] parallel_ops ({args.parallel_ops}) > eval_npu ({args.eval_npu}); "
            "multiple op jobs will share visible devices."
        )

    llm_config = get_llm_config_compatible(cli_model=args.model)
    resolved_model = llm_config["model"]
    model_slug = model_slug_for_path(resolved_model)
    out_run_dir = pathlib.Path(args.out_dir) if args.out_dir else _default_run_dir(
        model_slug=model_slug,
        tool_mode=args.tool_mode,
        strategy=args.strategy,
        run=args.run,
        test=args.test,
    )
    out_run_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Any] = {
        "model": resolved_model,
        "tool_mode": args.tool_mode,
        "strategy": args.strategy,
        "eval_mode": args.eval_mode,
        "clean_policy": args.clean_policy,
        "max_attempts": args.max_attempts,
        "parallel_ops": args.parallel_ops,
        "eval_workers": args.eval_workers,
        "eval_npu": args.eval_npu,
        "run_dir": str(out_run_dir),
        "ops": {},
    }

    if args.parallel_ops == 1:
        results = []
        for op_slot, op in enumerate(args.ops):
            category = dataset.get(op, {}).get("category", "activation")
            print(f"[RUN] op={op} category={category} strategy={args.strategy} tool_mode={args.tool_mode}")
            summary = _run_single_op_job(
                op=op,
                category=category,
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
            )
            results.append({"op": op, "summary": summary})
    else:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_ops) as ex:
            fut_to_op = {}
            for op_slot, op in enumerate(args.ops):
                category = dataset.get(op, {}).get("category", "activation")
                print(f"[RUN] op={op} category={category} strategy={args.strategy} tool_mode={args.tool_mode}")
                fut = ex.submit(
                    _run_single_op_job,
                    op=op,
                    category=category,
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
                )
                fut_to_op[fut] = op
            for fut in concurrent.futures.as_completed(fut_to_op):
                op = fut_to_op[fut]
                try:
                    results.append({"op": op, "summary": fut.result()})
                except Exception:
                    err_summary = {
                        "op": op,
                        "category": dataset.get(op, {}).get("category", "activation"),
                        "attempts": {},
                        "fixed_in_attempt2": False,
                        "fixed_on_attempt": None,
                        "max_attempts": args.max_attempts,
                        "final_status": "op_level_parallel_runtime_failed",
                        "error": traceback.format_exc(),
                    }
                    results.append({"op": op, "summary": err_summary})

    for item in sorted(results, key=lambda x: args.ops.index(x["op"])):
        op = item["op"]
        summary = item["summary"]
        all_summaries["ops"][op] = summary
        per_op_summary_path = out_run_dir / _per_op_summary_filename(op, args.max_attempts)
        _write_json(per_op_summary_path, summary)
        print(f"[WROTE] {per_op_summary_path}")

    aggregate_path = out_run_dir / _all_ops_summary_filename(args.max_attempts)
    _write_json(aggregate_path, all_summaries)
    print(f"[WROTE] {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
