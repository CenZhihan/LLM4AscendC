#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.agent.agent_builder import build_agent_app
from generator.agent.agent_config import get_llm_config_compatible, parse_tool_mode, tool_mode_to_string
from generator.agent.agent_state import create_initial_state
from generator.prompt_generators.prompt_utils import ascendc_template, read_relavant_files
from vendor.mkb.dataset import dataset


DEFAULT_TAG_PREFIX = "claude_20260427"
DEFAULT_SOC = "ai_core-Ascend910B2"

REQUESTED_TOOL_LABELS = {
    "kb": "kb_query",
    "code_search_snippet": "code_search_snippet",
    "code_rag": "code_rag",
    "api_lookup": "api_lookup",
    "api_constraint": "api_constraint",
    "tiling_calc": "tiling_calc",
    "tiling_validate": "tiling_validate",
    "npu_arch": "npu_arch",
    "env_check_env": "env_check",
    "env_check_npu": "env_check",
    "env_check_api": "env_check",
}

TOOL_DISPLAY_NAME = {
    "kb": "kb_query",
    "code_search_snippet": "code_search_snippet",
    "code_rag": "code_rag",
    "api_lookup": "api_lookup",
    "api_constraint": "api_constraint",
    "tiling_calc": "tiling_calc",
    "tiling_validate": "tiling_validate",
    "npu_arch": "npu_arch",
    "env_check_env": "env_check_env",
    "env_check_npu": "env_check_npu",
    "env_check_api": "env_check_api",
}

CATEGORY_TO_DIFFICULTY = {
    "activation": "simple",
    "broadcast": "medium",
    "reduce": "medium",
    "normalization": "medium",
    "matmul": "medium",
    "attention": "medium",
    "convolution": "complex",
    "fuse": "complex",
    "arch": "complex",
}


@dataclass(frozen=True)
class ExperimentConfig:
    tag: str
    tools: tuple[str, ...]
    purpose: str


DEFAULT_CASES = [
    {"op": "gelu", "category": "activation", "difficulty": "simple"},
    {"op": "add_bias_broadcast", "category": "broadcast", "difficulty": "medium"},
    {"op": "layer_norm", "category": "normalization", "difficulty": "medium"},
    {"op": "conv_pointwise_2d", "category": "convolution", "difficulty": "complex"},
]

DEFAULT_EXPERIMENTS = [
    ExperimentConfig("no_tool_baseline", tuple(), "Baseline without tool retrieval."),
    ExperimentConfig("kb_query_only", ("kb",), "Knowledge-base-only retrieval."),
    ExperimentConfig(
        "code_search_snippet_only",
        ("code_search_snippet",),
        "Structured code snippet retrieval only.",
    ),
    ExperimentConfig("code_rag_only", ("code_rag",), "Semantic code retrieval only."),
    ExperimentConfig(
        "api_lookup_constraint",
        ("api_lookup", "api_constraint"),
        "API signature plus constraint bundle.",
    ),
    ExperimentConfig(
        "tiling_calc_validate",
        ("tiling_calc", "tiling_validate"),
        "Tiling proposal plus validation bundle.",
    ),
    ExperimentConfig(
        "npu_arch_only",
        ("npu_arch",),
        "Hardware-architecture information only.",
    ),
    ExperimentConfig(
        "env_check_only",
        ("env_check_env", "env_check_npu", "env_check_api"),
        "Environment and installed-API checks only.",
    ),
    ExperimentConfig(
        "full_selected_tools",
        (
            "kb",
            "code_search_snippet",
            "code_rag",
            "api_lookup",
            "api_constraint",
            "tiling_calc",
            "tiling_validate",
            "npu_arch",
            "env_check_env",
            "env_check_npu",
            "env_check_api",
        ),
        "All requested tools enabled together.",
    ),
]


def _case_difficulty(category: str) -> str:
    return CATEGORY_TO_DIFFICULTY.get(category, "medium")


def _normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    op = str(case["op"])
    category = str(case.get("category") or dataset[op]["category"])
    difficulty = str(case.get("difficulty") or _case_difficulty(category))
    return {"op": op, "category": category, "difficulty": difficulty}


def _requested_tool_names(enabled_tools: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for tool in enabled_tools:
        label = REQUESTED_TOOL_LABELS.get(tool, tool)
        if label not in ordered:
            ordered.append(label)
    return ordered


def _tool_display_names(enabled_tools: tuple[str, ...]) -> list[str]:
    return [TOOL_DISPLAY_NAME.get(tool, tool) for tool in enabled_tools]


def _prefixed_tag(prefix: str, tag: str) -> str:
    return f"{prefix}_{tag}" if prefix else tag


def _build_one_shot_prompt(op: str) -> str:
    arch, example_arch, example_new_arch = read_relavant_files("ascendc", op, "leaky_relu")
    return ascendc_template(arch, example_arch, example_new_arch, op, "leaky_relu")


def _final_answer_from_state(final_state: dict[str, Any]) -> str:
    messages = final_state.get("messages") or []
    if not messages:
        return ""
    return getattr(messages[-1], "content", "") or ""


def _invoke_agent(op: str, category: str, base_prompt: str, tool_mode: tuple[str, ...]) -> dict[str, Any]:
    parsed_mode = parse_tool_mode(frozenset(tool_mode))
    llm_config = get_llm_config_compatible()
    app = build_agent_app(tool_mode=parsed_mode, llm_config=llm_config)
    initial_state = create_initial_state(
        base_prompt=base_prompt,
        op_name=op,
        category=category,
        language="ascendc",
        strategy_name="one_shot",
    )
    start = time.time()
    final_state = app.invoke(initial_state)
    duration_s = time.time() - start
    return {
        "duration_s": duration_s,
        "generation": _final_answer_from_state(final_state),
        "reasoning_content": final_state.get("reasoning_content", ""),
        "tool_calls": final_state.get("tool_calls_log", []),
        "tool_selection_trace": final_state.get("tool_choice_reasoning_log", []),
        "tool_choice_parse_errors": final_state.get("tool_choice_error_log", []),
        "tool_mode": tool_mode_to_string(parsed_mode),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_generation_txt(output_dir: Path, op: str, generation: str) -> Path:
    txt_path = output_dir / f"{op}.txt"
    txt_path.write_text(generation, encoding="utf-8")
    return txt_path


def _artifact_result_path(experiment_tag: str, op: str) -> Path:
    return REPO_ROOT / "artifacts" / "caseStudy" / experiment_tag / op / f"result_{op}.json"


def _parse_eval_result(experiment_tag: str, op: str) -> dict[str, Any]:
    result_path = _artifact_result_path(experiment_tag, op)
    if not result_path.is_file():
        return {
            "compiled": False,
            "precision_pass": False,
            "error_summary": f"Missing result json: {result_path}",
            "result_path": str(result_path),
            "logs": {},
        }

    payload = _load_json(result_path)
    op_result = ((payload.get("result") or {}).get(op) or {})
    meta = payload.get("meta") or {}
    return {
        "compiled": op_result.get("compiled"),
        "precision_pass": op_result.get("correctness"),
        "error_summary": op_result.get("correctness_info") or "",
        "result_path": str(result_path),
        "logs": meta.get("logs") or {},
    }


def _feedback_text(op: str, iter1_eval: dict[str, Any]) -> str:
    compiled = iter1_eval.get("compiled")
    precision_pass = iter1_eval.get("precision_pass")
    error_summary = (iter1_eval.get("error_summary") or "").strip()
    if compiled is True and precision_pass is True:
        status_line = "Iter 1 compiled and passed correctness. Preserve that behavior; only make conservative improvements if necessary."
    else:
        status_line = "Iter 1 did not fully pass evaluation. Repair the root cause described below."

    body = error_summary or "No detailed error summary was captured. Re-check API usage, tiling, naming, and pybind consistency."
    return textwrap.dedent(
        f"""
        Evaluation feedback for the previous attempt of `{op}`:
        - compiled: {compiled}
        - precision_pass: {precision_pass}
        - guidance: {status_line}
        - error_summary:
        {body}

        Regenerate the full six-string Python bundle. Follow the exact same output format. Fix only the issues required for compile/correctness success.
        """
    ).strip()


def _iter2_prompt(base_prompt: str, feedback: str) -> str:
    return f"{base_prompt}\n\n{feedback}\n"


def _run_eval_dir(output_dir: Path, workers: int, npu: int, label: str) -> tuple[int, Path]:
    log_path = output_dir / f"{label}_eval.log"
    cmd = [
        "python3",
        str(REPO_ROOT / "tools" / "eval_operator.py"),
        "--txt-dir",
        str(output_dir),
        "--workers",
        str(workers),
        "--clean-policy",
        "force",
        "--mode",
        "full",
    ]
    if workers > 1:
        cmd.extend(["--npu", str(npu)])

    env = os.environ.copy()
    env.setdefault("LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH", "/workspace/ascend_custom_opp")
    env.setdefault("LLM4ASCENDC_REF_ON_CPU", "1")

    eval_conda_env = env.get("LLM4ASCENDC_EVAL_CONDA_ENV", "").strip()
    if eval_conda_env:
        conda_sh = env.get("LLM4ASCENDC_CONDA_SH", "/root/miniconda3/etc/profile.d/conda.sh")
        ascend_env_sh = env.get("LLM4ASCENDC_ASCEND_ENV_SH", "/usr/local/Ascend/ascend-toolkit/set_env.sh")
        shell_cmd = " && ".join(
            [
                f"source {shlex.quote(conda_sh)}",
                f"conda activate {shlex.quote(eval_conda_env)}",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                f"source {shlex.quote(ascend_env_sh)}",
                " ".join(shlex.quote(part) for part in cmd),
            ]
        )
        proc = subprocess.run(
            ["bash", "-lc", shell_cmd],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(
            [sys.executable, *cmd[1:]],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_path.write_text(proc.stdout, encoding="utf-8")
    (output_dir / "eval.log").write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, log_path


def _empty_eval_summary() -> dict[str, Any]:
    return {"compiled": None, "precision_pass": None, "error_summary": "", "result_path": "", "logs": {}}


def _init_trace(config: ExperimentConfig, experiment_tag: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_tag": experiment_tag,
        "base_experiment_tag": config.tag,
        "enabled_tools": list(config.tools),
        "enabled_tool_labels": _tool_display_names(config.tools),
        "requested_tool_groups": _requested_tool_names(config.tools),
        "operator": case["op"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "iterations": {
            "iter1": {"tool_calls": [], "tool_selection_trace": [], "tool_choice_parse_errors": [], "generation": "", "eval": _empty_eval_summary()},
            "iter2": {"tool_calls": [], "tool_selection_trace": [], "tool_choice_parse_errors": [], "feedback_from_iter1": "", "generation": "", "eval": _empty_eval_summary()},
        },
    }


def _trace_path(output_dir: Path, op: str) -> Path:
    return output_dir / f"{op}_trace.json"


def _tool_round_count(iter_payload: dict[str, Any]) -> int:
    return len(iter_payload.get("tool_calls") or [])


def _rate(rows: list[dict[str, Any]], iteration: str, field: str) -> float:
    if not rows:
        return 0.0
    success = 0
    for row in rows:
        value = ((row.get("iterations") or {}).get(iteration) or {}).get("eval", {}).get(field)
        if value is True:
            success += 1
    return (100.0 * success) / len(rows)


def _format_bool(value: Any) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "N"
    return "-"


def _collect_issue_notes(rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    iter2_errors = [(((row.get("iterations") or {}).get("iter2") or {}).get("eval") or {}).get("error_summary") or "" for row in rows]
    joined = "\n".join(iter2_errors).lower()
    if "datacopy" in joined or "alignment" in joined:
        notes.append("DataCopy and alignment constraints remain a common failure mode.")
    if "tiling" in joined or "blockdim" in joined or "ub" in joined:
        notes.append("Tiling and block-dimension design still dominate medium/complex failures.")
    if "pybind" in joined or "module" in joined or "import" in joined:
        notes.append("Pybind packaging and module import consistency still need attention in failed cases.")
    if not notes:
        notes.append("Main residual failures are heterogeneous; inspect per-case error_summary fields for exact causes.")
    return notes


def _experiment_report_markdown(config: ExperimentConfig, experiment_tag: str, rows: list[dict[str, Any]], baseline_rates: dict[str, float] | None) -> str:
    cases_summary = ", ".join(f"{row['operator']}({row['difficulty']})" for row in rows)
    lines = [
        f"## Experiment Report: {experiment_tag}",
        "",
        "### Config",
        f"- Enabled tools: {_tool_display_names(config.tools) or ['no_tool']}",
        f"- Requested tool groups: {_requested_tool_names(config.tools) or ['baseline']}",
        f"- Test cases: [{cases_summary}]",
        "",
        "### Result Summary",
        "| Case | Difficulty | Iter1 Compiled | Iter1 Precision | Iter2 Compiled | Iter2 Precision | Tool Rounds |",
        "|------|------------|----------------|-----------------|----------------|-----------------|-------------|",
    ]
    for row in rows:
        iter1 = row["iterations"]["iter1"]
        iter2 = row["iterations"]["iter2"]
        lines.append(
            "| {case} | {difficulty} | {i1c} | {i1p} | {i2c} | {i2p} | {rounds} |".format(
                case=row["operator"],
                difficulty=row["difficulty"],
                i1c=_format_bool(iter1["eval"]["compiled"]),
                i1p=_format_bool(iter1["eval"]["precision_pass"]),
                i2c=_format_bool(iter2["eval"]["compiled"]),
                i2p=_format_bool(iter2["eval"]["precision_pass"]),
                rounds=_tool_round_count(iter1) + _tool_round_count(iter2),
            )
        )

    iter1_compile = _rate(rows, "iter1", "compiled")
    iter2_compile = _rate(rows, "iter2", "compiled")
    iter1_precision = _rate(rows, "iter1", "precision_pass")
    iter2_precision = _rate(rows, "iter2", "precision_pass")

    lines.extend([
        "",
        "### Compare To Baseline",
    ])
    if baseline_rates is None:
        lines.append("- Baseline experiment; no delta available.")
    else:
        lines.append(
            "- Compile pass rate delta: {:+.1f}% (Iter1), {:+.1f}% (Iter2)".format(
                iter1_compile - baseline_rates["iter1_compile_rate"],
                iter2_compile - baseline_rates["iter2_compile_rate"],
            )
        )
        lines.append(
            "- Precision pass rate delta: {:+.1f}% (Iter1), {:+.1f}% (Iter2)".format(
                iter1_precision - baseline_rates["iter1_precision_rate"],
                iter2_precision - baseline_rates["iter2_precision_rate"],
            )
        )

    lines.extend([
        "",
        "### Tool Effectiveness Notes",
    ])
    for note in _collect_issue_notes(rows):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _overall_summary_markdown(all_experiments: list[dict[str, Any]]) -> str:
    lines = [
        "## Cross-Experiment Summary",
        "",
        "| Experiment | Tools | Iter1 Compile | Iter1 Precision | Iter2 Compile | Iter2 Precision |",
        "|------------|-------|---------------|-----------------|---------------|-----------------|",
    ]
    for item in all_experiments:
        metrics = item["metrics"]
        tool_text = ", ".join(item["enabled_tool_labels"]) if item["enabled_tool_labels"] else "no_tool"
        lines.append(
            "| {tag} | {tools} | {i1c:.1f}% | {i1p:.1f}% | {i2c:.1f}% | {i2p:.1f}% |".format(
                tag=item["experiment_tag"],
                tools=tool_text,
                i1c=metrics["iter1_compile_rate"],
                i1p=metrics["iter1_precision_rate"],
                i2c=metrics["iter2_compile_rate"],
                i2p=metrics["iter2_precision_rate"],
            )
        )

    if all_experiments:
        baseline = all_experiments[0]["metrics"]
        best_compile = max(all_experiments[1:] or all_experiments, key=lambda item: item["metrics"]["iter2_compile_rate"])
        best_precision = max(all_experiments[1:] or all_experiments, key=lambda item: item["metrics"]["iter2_precision_rate"])
        lines.extend([
            "",
            "### Highlights",
            "- Best Iter2 compile rate: {tag} ({rate:.1f}%, baseline {base:.1f}%).".format(
                tag=best_compile["experiment_tag"],
                rate=best_compile["metrics"]["iter2_compile_rate"],
                base=baseline["iter2_compile_rate"],
            ),
            "- Best Iter2 precision rate: {tag} ({rate:.1f}%, baseline {base:.1f}%).".format(
                tag=best_precision["experiment_tag"],
                rate=best_precision["metrics"]["iter2_precision_rate"],
                base=baseline["iter2_precision_rate"],
            ),
        ])
    return "\n".join(lines) + "\n"


def _select_experiments(tags: list[str] | None) -> list[ExperimentConfig]:
    if not tags:
        return DEFAULT_EXPERIMENTS
    wanted = set(tags)
    selected = [config for config in DEFAULT_EXPERIMENTS if config.tag in wanted]
    missing = wanted - {config.tag for config in selected}
    if missing:
        raise ValueError(f"Unknown experiment tags: {sorted(missing)}")
    return selected


def _save_experiment_metadata(experiment_tag: str, config: ExperimentConfig, cases: list[dict[str, Any]], model_name: str) -> Path:
    output_dir = REPO_ROOT / "output" / "caseStudy" / experiment_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_tag": experiment_tag,
        "base_experiment_tag": config.tag,
        "enabled_tools": list(config.tools),
        "enabled_tool_labels": _tool_display_names(config.tools),
        "requested_tool_groups": _requested_tool_names(config.tools),
        "purpose": config.purpose,
        "model": model_name,
        "strategy": "one_shot",
        "test_cases": cases,
    }
    _write_json(output_dir / "experiment_metadata.json", metadata)
    return output_dir


def _update_trace_eval(output_dir: Path, op: str, iteration_key: str, eval_summary: dict[str, Any]) -> dict[str, Any]:
    trace_path = _trace_path(output_dir, op)
    trace = _load_json(trace_path)
    trace["iterations"][iteration_key]["eval"] = eval_summary
    _write_json(trace_path, trace)
    return trace


def _ensure_local_api_config() -> None:
    config_path = REPO_ROOT / "generator" / "local_api_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LLM config: {config_path}")


def run_experiments(*, cases: list[dict[str, Any]], experiments: list[ExperimentConfig], tag_prefix: str, workers: int, npu: int) -> list[dict[str, Any]]:
    _ensure_local_api_config()
    llm_model = get_llm_config_compatible()["model"]
    all_summaries: list[dict[str, Any]] = []
    baseline_rates: dict[str, float] | None = None

    for config in experiments:
        experiment_tag = _prefixed_tag(tag_prefix, config.tag)
        output_dir = _save_experiment_metadata(experiment_tag, config, cases, llm_model)
        print(f"[experiment] {experiment_tag} tools={list(config.tools) or ['no_tool']}")

        iter1_payloads: dict[str, dict[str, Any]] = {}
        for case in cases:
            op = case["op"]
            base_prompt = _build_one_shot_prompt(op)
            result = _invoke_agent(op, case["category"], base_prompt, config.tools)
            _save_generation_txt(output_dir, op, result["generation"])
            trace = _init_trace(config, experiment_tag, case)
            trace["iterations"]["iter1"].update(
                {
                    "tool_calls": result["tool_calls"],
                    "tool_selection_trace": result["tool_selection_trace"],
                    "tool_choice_parse_errors": result["tool_choice_parse_errors"],
                    "generation": result["generation"],
                    "duration_s": result["duration_s"],
                }
            )
            _write_json(_trace_path(output_dir, op), trace)
            iter1_payloads[op] = {"base_prompt": base_prompt, "result": result, "case": case}

        iter1_rc, iter1_log = _run_eval_dir(output_dir, workers=workers, npu=npu, label="iter1")
        print(f"[experiment] {experiment_tag} iter1 eval rc={iter1_rc} log={iter1_log}")

        iter2_inputs: dict[str, dict[str, Any]] = {}
        for case in cases:
            op = case["op"]
            iter1_eval = _parse_eval_result(experiment_tag, op)
            trace = _update_trace_eval(output_dir, op, "iter1", iter1_eval)
            feedback = _feedback_text(op, iter1_eval)
            iter2_prompt = _iter2_prompt(iter1_payloads[op]["base_prompt"], feedback)
            iter2_result = _invoke_agent(op, case["category"], iter2_prompt, config.tools)
            _save_generation_txt(output_dir, op, iter2_result["generation"])
            trace["iterations"]["iter2"].update(
                {
                    "tool_calls": iter2_result["tool_calls"],
                    "tool_selection_trace": iter2_result["tool_selection_trace"],
                    "tool_choice_parse_errors": iter2_result["tool_choice_parse_errors"],
                    "feedback_from_iter1": feedback,
                    "generation": iter2_result["generation"],
                    "duration_s": iter2_result["duration_s"],
                }
            )
            _write_json(_trace_path(output_dir, op), trace)
            iter2_inputs[op] = {"case": case, "trace": trace}

        iter2_rc, iter2_log = _run_eval_dir(output_dir, workers=workers, npu=npu, label="iter2")
        print(f"[experiment] {experiment_tag} iter2 eval rc={iter2_rc} log={iter2_log}")

        experiment_rows: list[dict[str, Any]] = []
        for case in cases:
            op = case["op"]
            trace = _update_trace_eval(output_dir, op, "iter2", _parse_eval_result(experiment_tag, op))
            experiment_rows.append(trace)

        report_md = _experiment_report_markdown(config, experiment_tag, experiment_rows, baseline_rates)
        (output_dir / "report.md").write_text(report_md, encoding="utf-8")

        metrics = {
            "iter1_compile_rate": _rate(experiment_rows, "iter1", "compiled"),
            "iter1_precision_rate": _rate(experiment_rows, "iter1", "precision_pass"),
            "iter2_compile_rate": _rate(experiment_rows, "iter2", "compiled"),
            "iter2_precision_rate": _rate(experiment_rows, "iter2", "precision_pass"),
        }
        summary = {
            "experiment_tag": experiment_tag,
            "base_experiment_tag": config.tag,
            "enabled_tools": list(config.tools),
            "enabled_tool_labels": _tool_display_names(config.tools),
            "requested_tool_groups": _requested_tool_names(config.tools),
            "purpose": config.purpose,
            "metrics": metrics,
            "cases": experiment_rows,
            "logs": {"iter1": str(iter1_log), "iter2": str(iter2_log)},
            "report_path": str(output_dir / "report.md"),
        }
        _write_json(output_dir / "summary.json", summary)
        all_summaries.append(summary)

        if baseline_rates is None:
            baseline_rates = metrics

    return all_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled tool-ablation case studies for AscendC agent generation.")
    parser.add_argument("--tag-prefix", default=DEFAULT_TAG_PREFIX, help="Prefix added to every experiment tag.")
    parser.add_argument("--workers", type=int, default=4, help="Workers passed to eval_operator.py --txt-dir.")
    parser.add_argument("--npu", type=int, default=2, help="NPU count passed to eval_operator.py when workers > 1.")
    parser.add_argument("--experiments", nargs="*", help="Optional subset of experiment base tags to run.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 1 and args.npu < 1:
        raise ValueError("--npu must be >= 1 when --workers > 1")

    cases = [_normalize_case(case) for case in DEFAULT_CASES]
    experiments = _select_experiments(args.experiments)
    summaries = run_experiments(
        cases=cases,
        experiments=experiments,
        tag_prefix=args.tag_prefix,
        workers=args.workers,
        npu=args.npu,
    )

    summary_root = REPO_ROOT / "output" / "caseStudy"
    overall_json = summary_root / f"{args.tag_prefix}_summary.json"
    overall_md = summary_root / f"{args.tag_prefix}_summary.md"
    _write_json(overall_json, {"tag_prefix": args.tag_prefix, "cases": cases, "experiments": summaries, "soc": DEFAULT_SOC})
    overall_md.write_text(_overall_summary_markdown(summaries), encoding="utf-8")

    print(f"[done] wrote {overall_json}")
    print(f"[done] wrote {overall_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
