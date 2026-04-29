#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _run_eval_for_txt(txt_path: pathlib.Path, mode: str, clean_policy: str) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "eval_operator.py"),
        "--txt",
        str(txt_path),
        "--mode",
        mode,
        "--clean-policy",
        clean_policy,
    ]
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


def run_two_round_for_op(
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
) -> Dict[str, Any]:
    attempt1_dir = run_dir / "attempt1"
    attempt2_dir = run_dir / "attempt2"
    op_summary: Dict[str, Any] = {
        "op": op,
        "category": category,
        "attempts": {},
        "fixed_in_attempt2": False,
        "final_status": "unknown",
    }

    # Attempt 1
    outcome1 = AttemptOutcome(
        attempt_id=1,
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
        gen1 = _generate_one_attempt(
            op=op,
            category=category,
            strategy=strategy,
            tool_mode=tool_mode,
            llm_config=llm_config,
            out_dir=attempt1_dir,
            attempt_id=1,
        )
        outcome1.generated = True
        outcome1.txt_path = str(gen1["txt_path"])
        outcome1.report_path = str(gen1["report_path"])
        eval_rc1 = _run_eval_for_txt(gen1["txt_path"], mode=eval_mode, clean_policy=clean_policy)
        outcome1.eval_ran = True
        result1_path = _eval_result_json_path(gen1["txt_path"], op)
        if not result1_path.exists():
            raise RuntimeError(
                f"eval finished (rc={eval_rc1}) but result json not found: {result1_path}"
            )
        outcome1.eval_result_path = str(result1_path)
        payload1 = _load_result_payload(result1_path)
        core1 = _extract_eval_core(payload1, op)
        outcome1.compiled = core1["compiled"]
        outcome1.correctness = core1["correctness"]
        outcome1.correctness_info = core1["correctness_info"]
        outcome1.selected_log_paths = _select_error_logs(core1["logs"])
        repair_text = _build_repair_error_context(
            op=op,
            result_payload=payload1,
            selected_logs=outcome1.selected_log_paths,
            max_log_lines=max_log_lines,
        )
        repair_path = attempt1_dir / f"{op}_repair_context.txt"
        _write_text(repair_path, repair_text)
        outcome1.repair_context_path = str(repair_path)
        attempt1_generated_code = gen1["result"].generated_code
    except Exception:
        outcome1.generation_error = traceback.format_exc()
        attempt1_generated_code = ""

    op_summary["attempts"]["attempt1"] = outcome1.__dict__

    # Attempt 2 only when attempt1 evaluated and failed
    need_attempt2 = (
        outcome1.generated
        and outcome1.eval_ran
        and _attempt_failed(outcome1.compiled, outcome1.correctness)
    )
    if not need_attempt2:
        if outcome1.generated and outcome1.eval_ran and not _attempt_failed(outcome1.compiled, outcome1.correctness):
            op_summary["final_status"] = "pass_on_attempt1"
        elif outcome1.generation_error:
            op_summary["final_status"] = "generation_failed_on_attempt1"
        else:
            op_summary["final_status"] = "attempt1_failed_without_attempt2"
        return op_summary

    outcome2 = AttemptOutcome(
        attempt_id=2,
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
        repair_logs_raw = pathlib.Path(outcome1.repair_context_path).read_text(
            encoding="utf-8", errors="replace"
        )
        gen2 = _generate_one_attempt(
            op=op,
            category=category,
            strategy=strategy,
            tool_mode=tool_mode,
            llm_config=llm_config,
            out_dir=attempt2_dir,
            attempt_id=2,
            repair_error_logs_raw=repair_logs_raw,
            previous_attempt_code=attempt1_generated_code,
        )
        outcome2.generated = True
        outcome2.txt_path = str(gen2["txt_path"])
        outcome2.report_path = str(gen2["report_path"])
        eval_rc2 = _run_eval_for_txt(gen2["txt_path"], mode=eval_mode, clean_policy=clean_policy)
        outcome2.eval_ran = True
        result2_path = _eval_result_json_path(gen2["txt_path"], op)
        if not result2_path.exists():
            raise RuntimeError(
                f"eval finished (rc={eval_rc2}) but result json not found: {result2_path}"
            )
        outcome2.eval_result_path = str(result2_path)
        payload2 = _load_result_payload(result2_path)
        core2 = _extract_eval_core(payload2, op)
        outcome2.compiled = core2["compiled"]
        outcome2.correctness = core2["correctness"]
        outcome2.correctness_info = core2["correctness_info"]
        outcome2.selected_log_paths = _select_error_logs(core2["logs"])
        if not _attempt_failed(outcome2.compiled, outcome2.correctness):
            op_summary["fixed_in_attempt2"] = True
            op_summary["final_status"] = "fixed_on_attempt2"
        else:
            op_summary["final_status"] = "failed_after_attempt2"
    except Exception:
        outcome2.generation_error = traceback.format_exc()
        op_summary["final_status"] = "attempt2_generation_or_eval_failed"

    op_summary["attempts"]["attempt2"] = outcome2.__dict__
    return op_summary


def _default_run_dir(*, model_slug: str, tool_mode: str, strategy: str, run: int, test: bool) -> pathlib.Path:
    from generator.agent.agent_config import parse_tool_mode, tool_mode_to_string

    out_root = pathlib.Path("output/test/ascendc" if test else "output/ascendc")
    return out_root / model_slug / f"agent_{tool_mode_to_string(parse_tool_mode(tool_mode))}" / strategy / f"run{run}"


def main() -> int:
    from vendor.mkb.dataset import dataset
    from generator.agent.agent_config import get_llm_config_compatible, model_slug_for_path

    parser = argparse.ArgumentParser(description="Two-round automated agent generation + eval + repair.")
    parser.add_argument("--ops", nargs="+", required=True, help="Operator keys to run.")
    parser.add_argument("--tool-mode", type=str, default="all", help="Tool mode string.")
    parser.add_argument("--strategy", type=str, default="one_shot", help="Prompt strategy.")
    parser.add_argument("--run", type=int, default=0, help="Run index for output path.")
    parser.add_argument("--out-dir", type=str, default="", help="Custom run directory (contains attempt1/attempt2).")
    parser.add_argument("--test", action="store_true", help="Use output/test/ascendc when --out-dir is not set.")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model.")
    parser.add_argument("--eval-mode", choices=["full", "build-only", "eval-only"], default="full")
    parser.add_argument("--clean-policy", choices=["force", "smart"], default="force")
    parser.add_argument("--max-log-lines", type=int, default=220, help="Tail lines per selected log in repair context.")
    args = parser.parse_args()

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
        "run_dir": str(out_run_dir),
        "ops": {},
    }

    for op in args.ops:
        category = dataset.get(op, {}).get("category", "activation")
        print(f"[RUN] op={op} category={category} strategy={args.strategy} tool_mode={args.tool_mode}")
        summary = run_two_round_for_op(
            op=op,
            category=category,
            strategy=args.strategy,
            tool_mode=args.tool_mode,
            llm_config=llm_config,
            run_dir=out_run_dir,
            eval_mode=args.eval_mode,
            clean_policy=args.clean_policy,
            max_log_lines=args.max_log_lines,
        )
        all_summaries["ops"][op] = summary
        per_op_summary_path = out_run_dir / f"{op}_two_round_summary.json"
        _write_json(per_op_summary_path, summary)
        print(f"[WROTE] {per_op_summary_path}")

    aggregate_path = out_run_dir / "two_round_summary_all_ops.json"
    _write_json(aggregate_path, all_summaries)
    print(f"[WROTE] {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
