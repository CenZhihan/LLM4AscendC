from __future__ import annotations

# 使用智能体生成算子 Kernel（支持 KB、Web、Code RAG 多源检索）
import os
import sys

# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from generator.agent import (
    generate_kernel_with_agent,
    KernelGenerationTask,
)

from generator.agent.agent_config import (
    parse_tool_mode,
    AgentToolMode,
    has_code_rag,
    tool_mode_to_string,
    get_llm_config_compatible,
    model_slug_for_path,
)

from generator.agent.retrievers.code_retriever import CodeRetriever
from generator.dataset import dataset
from generator.config import rag_index_path, rag_embedding_model


def _load_code_retriever():
    """Pre-load Code RAG retriever for parallel generation."""
    retriever = CodeRetriever(devices=['cpu'])
    if retriever.is_available():
        print(f"[INFO] Code RAG retriever loaded")
        return retriever
    else:
        print("[WARN] Failed to load Code RAG index")
        return None


def _generate_one_op(
    op: str,
    category: str,
    out_dir: str,
    tool_mode: AgentToolMode,
    code_retriever: CodeRetriever,
    strategy: str,
    llm_config: dict,
):
    """Single operator generation task."""
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[SKIP] {op} already exists")
        return op, None

    try:
        print(f"[START] Generating kernel for {op} (category: {category})")

        task = KernelGenerationTask(
            language="ascendc",
            op=op,
            strategy_name=strategy,
            category=category,
        )

        result = generate_kernel_with_agent(
            task,
            tool_mode=tool_mode,
            retriever=code_retriever,
            llm_config=llm_config,
        )

        # Write output
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(result.generated_code)

        # Write reasoning if available
        if result.reasoning:
            with open(os.path.join(out_dir, f'{op}_cot.txt'), 'w', encoding='utf-8') as f:
                f.write(result.reasoning)

        # Write report if available
        if result.report:
            import json
            with open(os.path.join(out_dir, f'{op}_report.json'), 'w', encoding='utf-8') as f:
                json.dump(result.report, f, ensure_ascii=False, indent=2)

        print(f"[DONE] {op}")
        return op, None

    except Exception as e:
        print(f"[FAIL] {op}: {e}")
        import traceback
        traceback.print_exc()
        return op, e


def main():
    parser = argparse.ArgumentParser(description="使用智能体生成算子 Kernel（支持多源检索）")

    parser.add_argument(
        "--tool-mode",
        type=str,
        default="no_tool",
        help=(
            "检索工具模式：预置名 no_tool/kb_only/web_only/.../all，"
            "或逗号分隔列表如 kb,web,code_rag,code_search_snippet 及已注册的插件名（如 kb,my_plugin）。"
        ),

    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="add_shot",
        help="Prompt 策略名称 (默认: add_shot)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行工作数 (默认: 4)"
    )
    parser.add_argument(
        "--categories",
        nargs='+',
        default=['all'],
        help="算子类别列表 (默认: all)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="运行次数 (默认: 1)"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="从指定算子开始（断点续传）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="自定义输出目录"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖配置文件中的模型名（优先级高于 generator/local_api_config.py 中的 XI_AI_MODEL）",
    )
    parser.add_argument(
        "--kernelbench102",
        action="store_true",
        help="只生成 kernelbench102 中的 102 个算子"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：默认输出根目录改为 output/test/ascendc（仅在未传 --output-dir 时生效）",
    )

    args = parser.parse_args()

    try:
        llm_config = get_llm_config_compatible(cli_model=args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] LLM 配置: {e}")
        raise SystemExit(2) from e

    resolved_model = llm_config["model"]
    model_slug = model_slug_for_path(resolved_model)
    print(f"[INFO] LLM model: {resolved_model} (output segment: {model_slug})")

    try:
        tool_mode = parse_tool_mode(args.tool_mode)
    except ValueError as e:
        print(f"[ERROR] Invalid --tool-mode: {e}")
        raise SystemExit(2) from e
    strategy = args.strategy
    workers = args.workers

    tool_labels = [str(t) for t in sorted(tool_mode)]
    print(f"[INFO] Tool mode: {', '.join(tool_labels)}")
    print(f"[INFO] Strategy: {strategy}")
    print(f"[INFO] Workers: {workers}")
    print(f"[INFO] Categories: {args.categories}")

    # Get operator list
    all_ops = list(dataset.keys())
    if args.categories != ['all']:
        all_ops = [op for op in all_ops if dataset[op]['category'] in args.categories]

    if args.kernelbench102:
        from generator.kernelbench102_ops import KERNELBENCH102_OP_SET
        all_ops = [op for op in all_ops if op in KERNELBENCH102_OP_SET]
        print(f"[INFO] KernelBench102 filter: {len(all_ops)} ops")

    all_ops = sorted(all_ops)
    print(f"[INFO] Total ops to generate: {len(all_ops)}")

    # Pre-load Code RAG retriever if needed
    code_retriever = None
    if has_code_rag(tool_mode):
        print("[INFO] Loading Code RAG retriever...")
        code_retriever = _load_code_retriever()

    # Handle start-from
    start_index = 0
    if args.start_from:
        if args.start_from in all_ops:
            start_index = all_ops.index(args.start_from)
            print(f"[INFO] Starting from op: {args.start_from} (index {start_index})")
        else:
            print(f"[WARN] Start op '{args.start_from}' not found, starting from beginning")

    for run in range(args.runs):
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_root = "output/test/ascendc" if args.test else "output/ascendc"
            out_dir = (
                f"{out_root}/{model_slug}/agent_{tool_mode_to_string(tool_mode)}"
                f"/{strategy}/run{run}"
            )

        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Output directory: {out_dir}")

        ops_to_process = all_ops[start_index:]

        # Parallel generation
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_op, op, dataset[op]['category'],
                    out_dir, tool_mode, code_retriever, strategy, llm_config
                ): op
                for op in ops_to_process
            }

            for future in as_completed(futures):
                op, err = future.result()
                if err is None:
                    success_count += 1
                else:
                    fail_count += 1

        print(f"\n[SUMMARY] Run {run + 1} completed: {success_count} success, {fail_count} failed")

    print("[INFO] All done.")


if __name__ == "__main__":
    main()