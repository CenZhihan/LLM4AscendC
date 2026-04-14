#!/usr/bin/env python3
"""
Generate MKB-style AscendC operator bundles (*.txt) for the 102 kernelbench165 passed ops only.

Requires: XI_AI_API_KEY (and optional XI_AI_BASE_URL, XI_AI_MODEL).
Does not run eval_operator; use tools/eval_operator.py --txt on outputs.

Repo root must be on PYTHONPATH (run from repo root as below).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generation.prompt_generators  # noqa: F401 — registers ascendc none / one_shot

from generation import gen_config
from generation.agent.agent_config import AgentToolMode
from generation.direct_generate import generate_and_write_single
from generation.kernelbench102_ops import KERNELBENCH102_OP_KEYS
from generation.llm_config import get_xi_model_name, get_xi_openai_client
from generation.prompt_generators.prompt_registry import PROMPT_REGISTRY
from vendor.mkb import dataset as mkb_dataset


def _resolve_ops(categories: list[str]) -> list[str]:
    all_mode = categories == ["all"]
    cats = set(categories)
    out: list[str] = []
    for op in KERNELBENCH102_OP_KEYS:
        if op not in mkb_dataset.dataset:
            print(f"[WARN] {op} not in vendor/mkb/dataset.py, skip")
            continue
        cat = mkb_dataset.dataset[op]["category"]
        if all_mode or cat in cats:
            out.append(op)
    return out


def generate_prompt(language: str, strategy_name: str, op: str) -> str:
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        raise ValueError(f"Unknown prompt strategy: {language}/{strategy_name}")
    return PROMPT_REGISTRY[language][strategy_name].generate(op)


def _agent_output_dir_base(
    language: str,
    base_strategy: str,
    model_name: str,
    tool_mode: AgentToolMode,
) -> str:
    tools_name = {
        AgentToolMode.NO_TOOL: "no_tool",
        AgentToolMode.KB_ONLY: "kb_only",
        AgentToolMode.WEB_ONLY: "web_only",
        AgentToolMode.KB_AND_WEB: "kb_and_web",
    }[tool_mode]
    strategy_dir = f"agent_{base_strategy}_tools={tools_name}"
    return f"output/{language}/{strategy_dir}/{gen_config.temperature}-{gen_config.top_p}/{model_name}"


def _generate_one_direct(op: str, out_dir: str, strategy: str, model: str) -> tuple[str, Exception | None]:
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[INFO] Already generated {out_path}, skip")
        return op, None
    try:
        print(f"[INFO] Generate {op} strategy={strategy}")
        prompt = generate_prompt("ascendc", strategy, op)
        client = get_xi_openai_client()
        m = model or get_xi_model_name()
        generate_and_write_single(prompt, client, out_dir, op, model=m)
        print(f"[INFO] Done {op}")
        return op, None
    except Exception as e:
        print(f"[FAIL] {op}: {e}")
        return op, e


def _generate_one_agent(
    op: str,
    strategy: str,
    tool_mode: AgentToolMode,
    model_name: str,
    run: int,
) -> tuple[str, Exception | None]:
    from generation.agent.agent_runner import AgentResult, KernelTask, generate_kernel_with_agent

    base = _agent_output_dir_base("ascendc", strategy, model_name, tool_mode)
    out_dir = os.path.join(ROOT, base, f"run{run}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[INFO][Agent] Already generated {out_path}, skip")
        return op, None
    try:
        print(f"[INFO][Agent] {op} strategy={strategy} tools={tool_mode.value}")
        task = KernelTask(language="ascendc", op=op, strategy_name=strategy)
        result: AgentResult = generate_kernel_with_agent(task=task, tool_mode=tool_mode)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.raw_answer)
        if result.report:
            report_path = os.path.join(out_dir, f"{op}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(result.report, f, indent=2, ensure_ascii=False)
        print(f"[INFO][Agent] Done {op}")
        return op, None
    except Exception as e:
        print(f"[FAIL][Agent] {op}: {e}")
        return op, e


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AscendC operator txt for kernelbench102 subset.")
    parser.add_argument("--runs", type=int, default=1, help="Number of run directories (run0, run1, ...).")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="OpenAI API model name (default: XI_AI_MODEL env or generation llm_config default).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["none", "one_shot"],
        default="one_shot",
        help="Prompt strategy (only none or one_shot).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="MKB categories to include, or 'all' for all 102 ops.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Thread pool size (direct mode).")
    parser.add_argument("--use_agent", action="store_true", help="Use LangGraph agent instead of single-shot chat.")
    parser.add_argument(
        "--agent_tools",
        type=str,
        default="no_tool",
        choices=[m.value for m in AgentToolMode],
        help="Agent tool mode when --use_agent.",
    )
    parser.add_argument(
        "--agent_workers",
        type=int,
        default=4,
        help="Thread pool size when --use_agent.",
    )
    args = parser.parse_args()

    model_display = args.model or get_xi_model_name()
    if "/" in model_display:
        model_name = model_display.split("/")[-1]
    else:
        model_name = model_display

    op_list = _resolve_ops(args.categories)
    if not op_list:
        print("No operators to generate (check --categories).")
        return 1

    print(f"Operators to generate: {len(op_list)} (kernelbench102 subset)")
    print(f"Strategy: {args.strategy}, model: {model_display}, use_agent={args.use_agent}")

    rc = 0
    for run in range(args.runs):
        if not args.use_agent:
            out_dir = os.path.join(
                ROOT,
                f"output/ascendc/{args.strategy}/{gen_config.temperature}-{gen_config.top_p}/{model_name}/run{run}",
            )
            os.makedirs(out_dir, exist_ok=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(_generate_one_direct, op, out_dir, args.strategy, args.model): op
                    for op in op_list
                }
                for fut in as_completed(futs):
                    _op, err = fut.result()
                    if err is not None:
                        rc = 1
        else:
            tool_mode = AgentToolMode(args.agent_tools)
            with ThreadPoolExecutor(max_workers=args.agent_workers) as ex:
                futs = {
                    ex.submit(
                        _generate_one_agent,
                        op,
                        args.strategy,
                        tool_mode,
                        model_name,
                        run,
                    ): op
                    for op in op_list
                }
                for fut in as_completed(futs):
                    _op, err = fut.result()
                    if err is not None:
                        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
