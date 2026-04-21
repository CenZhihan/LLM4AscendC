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
    p.add_argument("--tool_mode", default="all", help="Tool mode string (e.g. all, kb,web,code_rag)")
    p.add_argument("--strategy", default="one_shot", help="Prompt strategy name (one_shot, none, etc.)")
    p.add_argument("--outdir", default="output/agent_runs", help="Directory to write per-op JSON reports")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for op in args.ops:
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


if __name__ == "__main__":
    main()
