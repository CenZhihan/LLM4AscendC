#!/usr/bin/env python3
"""
Run agent experiments and record tool-call chains.

Usage:
  python generator/scripts/run_agent_experiments.py --ops gelu matmul_gelu_softmax --tool_mode all --strategy one_shot

Notes:
- Requires generator/local_api_config.py to exist (LLM credentials + model).
- Install dependencies from `requirements.txt` if needed (langgraph, langchain-core, openai, etc.).
"""

import argparse
import json
import os
import time
import traceback

from vendor.mkb.dataset import dataset
from generator.agent.agent_runner import KernelGenerationTask, generate_kernel_with_agent


CASE_STUDY_OPS = [
    "conv_pointwise_2d",
    "gelu",
    "relu",
]


def run_one(op: str, tool_mode: str, strategy: str):
    category = dataset.get(op, {}).get("category", "activation")
    task = KernelGenerationTask(language="ascendc", op=op, strategy_name=strategy, category=category)
    start = time.time()
    result = generate_kernel_with_agent(task, tool_mode)
    dur = time.time() - start
    return result, dur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="+", default=["gelu", "matmul_gelu_softmax"], help="Operator keys to generate")
    p.add_argument("--tool_mode", default="all", help="Tool mode string (e.g. all, kb,web,code_rag,code_search_snippet or code_search_snippet)")
    p.add_argument("--strategy", default="one_shot", help="Prompt strategy name (one_shot, none, etc.)")
    p.add_argument("--outdir", default="output/agent_runs", help="Directory to write per-op JSON reports")
    p.add_argument(
        "--case-study",
        action="store_true",
        help="Run CASE_STUDY_OPS (default: conv_pointwise_2d, gelu, relu)",
    )
    args = p.parse_args()

    ops = CASE_STUDY_OPS if args.case_study else args.ops

    os.makedirs(args.outdir, exist_ok=True)

    for op in ops:
        print(f"[RUN] op={op} tool_mode={args.tool_mode} strategy={args.strategy}")
        try:
            res, dur = run_one(op, args.tool_mode, args.strategy)
            out = {
                "op": res.op,
                "duration_s": dur,
                "generated_code": res.generated_code,
                "reasoning": res.reasoning,
                "tool_usage": res.tool_usage,
                "report": res.report,
            }
        except Exception:
            out = {"op": op, "error": traceback.format_exc()}

        out_path = os.path.join(args.outdir, f"{op}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[WROTE] {out_path}")

        generated_code = out.get("generated_code")
        if isinstance(generated_code, str) and generated_code.strip():
            txt_path = os.path.join(args.outdir, f"{op}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(generated_code)
            print(f"[WROTE] {txt_path}")


if __name__ == "__main__":
    main()
