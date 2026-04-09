# RAG 算子生成：使用代码 Embedding RAG 检索相关代码片段，支持并行生成
import os
import sys
# 添加项目根目录到 sys.path（支持从 generator/scripts 执行）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from generator.utils import get_client, get_default_model_from_config
from generator.config import (
    temperature, num_completions, top_p,
    rag_index_path, rag_embedding_model, rag_top_k, rag_max_chars,
    project_root_path, ref_impl_base_path
)
from generator.prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from generator.scripts.generation.generate_and_write import generate_and_write_single
from generator.rag import EmbeddingRetriever
from generator.dataset import dataset, category2exampleop


def _load_code_rag_retriever():
    """加载代码 Embedding RAG 检索器"""
    retriever = EmbeddingRetriever(
        index_path=rag_index_path,
        model_name=rag_embedding_model,
        devices=['cpu']  # 强制使用 CPU，避免 NPU 初始化问题
    )
    if not retriever.load_index():
        print("[WARN] Failed to load code RAG index, retriever will return empty results")
    return retriever


def _extract_key_info_from_code(op: str, category: str) -> dict:
    """从算子参考代码中提取关键信息，用于构建更精准的查询"""
    ref_path = os.path.join(ref_impl_base_path, f"{category}/{op}.py")
    key_info = {
        'op_name': op,
        'category': category,
        'forward_logic': '',
        'tensor_ops': [],
        'api_keywords': []
    }

    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # 提取 forward 函数体
        forward_match = re.search(r'def forward\([^)]*\)[^:]*:\s*(.+?)(?=\n    def |\nclass |\Z)', code, re.DOTALL)
        if forward_match:
            forward_code = forward_match.group(1).strip()
            key_info['forward_logic'] = forward_code[:200]

        # 提取 torch 操作关键字
        torch_ops = re.findall(r'torch\.(\w+)', code)
        tensor_ops = re.findall(r'\.(\w+)\(', code)
        key_info['tensor_ops'] = list(set(torch_ops + tensor_ops))[:10]

        # 类别相关的 API 关键词
        category_api_map = {
            "activation": ["Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "LeakyRelu", "Swish", "activation"],
            "matmul": ["MatMul", "BatchMatMul", "Gemm", "matrix multiplication", "Linear"],
            "convolution": ["Conv2D", "Conv3D", "ConvTranspose", "convolution", "depthwise"],
            "attention": ["Attention", "Softmax", "Query", "Key", "Value", "scaled_dot_product"],
            "normalization": ["LayerNorm", "BatchNorm", "InstanceNorm", "GroupNorm", "RmsNorm"],
            "pooling": ["MaxPool", "AvgPool", "Pool2D", "Pool3D", "adaptive_pool"],
            "loss": ["CrossEntropy", "MSELoss", "KLDivLoss", "NLLLoss", "loss function"],
            "reduce": ["ReduceSum", "ReduceMax", "ReduceMin", "ReduceMean", "reduction"],
            "broadcast": ["Broadcast", "elementwise", "Add", "Mul", "Sub", "Div"],
            "math": ["CumSum", "CumProd", "Exp", "Log", "Sqrt", "Pow"],
            "index": ["Gather", "Scatter", "IndexSelect", "Embedding", "ArgMax", "ArgMin"],
            "resize": ["Upsample", "Interpolate", "Resize", "GridSample", "bilinear"],
            "fuse": ["fused kernel", "fusion", "composite"],
            "arch": ["ResNet", "VGG", "Transformer", "LSTM", "GRU", "block"],
        }
        key_info['api_keywords'] = category_api_map.get(category, [])

    except Exception as e:
        print(f"[WARN] Failed to extract key info for {op}: {e}")

    return key_info


def _build_rag_query(op: str, category: str, key_info: dict = None) -> str:
    """构建 RAG 查询语句"""
    if key_info is None:
        key_info = _extract_key_info_from_code(op, category)

    query_parts = []

    # Level 1: 核心算子名称 + 类别
    query_parts.append(f"AscendC {op} kernel implementation")

    # Level 2: 类别相关 API 关键词
    if key_info.get('api_keywords'):
        api_keywords = key_info['api_keywords'][:3]
        query_parts.append(f"API: {' '.join(api_keywords)}")

    # Level 3: forward 逻辑摘要
    if key_info.get('forward_logic'):
        forward_snippet = key_info['forward_logic'][:100]
        query_parts.append(f"operation: {forward_snippet}")

    # Level 4: tensor 操作
    if key_info.get('tensor_ops'):
        ops_str = ' '.join(key_info['tensor_ops'][:5])
        query_parts.append(f"torch ops: {ops_str}")

    return '\n'.join(query_parts)


def _format_retrieved_code(results: list[dict], max_chars: int) -> str:
    """格式化检索到的代码片段"""
    if not results:
        return ""

    formatted_parts = []
    total_chars = 0

    for i, result in enumerate(results):
        code = result['code']
        file_path = result.get('file', 'unknown')
        score = result.get('score', 0.0)

        header = f"### Retrieved Code {i+1} (score: {score:.3f}, file: {file_path})\n\n"
        code_block = f"```cpp\n{code}\n```\n\n"

        part = header + code_block
        if total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                formatted_parts.append(part[:remaining] + "\n\n(... truncated ...)")
            break

        formatted_parts.append(part)
        total_chars += len(part)

    return "".join(formatted_parts)


def generate_prompt_rag_code(op: str, category: str, retriever: EmbeddingRetriever):
    """使用代码 RAG 检索相关代码，拼入 prompt 生成"""
    # 提取关键信息
    key_info = _extract_key_info_from_code(op, category)

    # 获取基础的 ascendc prompt
    arc_src, example_arch_src, example_new_arch_src = read_relavant_files("ascendc", op, "add")
    base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, "add")

    # 构建查询并检索相关代码
    query = _build_rag_query(op, category, key_info)
    results = retriever.retrieve(query, top_k=rag_top_k)

    if not results:
        print(f"[WARN] No code retrieved for op {op}")
        return base_prompt

    # 格式化检索结果
    retrieved_code = _format_retrieved_code(results, rag_max_chars)

    if not retrieved_code:
        return base_prompt

    # 拼接 RAG 代码片段到 prompt 开头
    rag_section = "## Retrieved AscendC Code References\n\n"
    rag_section += "The following code snippets were retrieved from the codebase and may be helpful:\n\n"
    rag_section += retrieved_code
    rag_section += "---\n\n"

    return rag_section + base_prompt


def _generate_one_op(op: str, category: str, out_dir: str, retriever: EmbeddingRetriever, model: str):
    """单个算子生成任务"""
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[SKIP] {op} already exists")
        return op, None

    try:
        print(f"[START] Generating kernel for {op} (category: {category})")
        prompt = generate_prompt_rag_code(op, category, retriever)
        client = get_client(model)
        generate_and_write_single(prompt, client, out_dir, op, model)
        print(f"[DONE] {op}")
        return op, None
    except Exception as e:
        print(f"[FAIL] {op}: {e}")
        return op, e


def main():
    parser = argparse.ArgumentParser(description="使用 RAG 代码检索生成算子 Kernel (并行版本)")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument("--model", type=str, default=_default_model, help="模型名称")
    parser.add_argument("--runs", type=int, default=1, help="运行次数")
    parser.add_argument("--top-k", type=int, default=rag_top_k, help="检索 top_k")
    parser.add_argument("--max-chars", type=int, default=rag_max_chars, help="检索内容最大字符数")
    parser.add_argument("--categories", nargs='+', default=['all'], help='算子类别')
    parser.add_argument("--workers", type=int, default=4, help="并行工作数")
    parser.add_argument("--start-from", type=str, default=None, help="从指定算子开始（断点续传）")
    args = parser.parse_args()

    model = args.model
    runs = args.runs
    workers = args.workers
    model_name = model.split("/")[-1] if "/" in model else model

    # 获取算子列表
    all_ops = list(dataset.keys())

    # 按 category 过滤
    if args.categories != ['all']:
        all_ops = [op for op in all_ops if dataset[op]['category'] in args.categories]

    # 排序
    all_ops = sorted(all_ops)

    print(f"[INFO] Total ops to generate: {len(all_ops)}")
    print(f"[INFO] Categories: {args.categories}")
    print(f"[INFO] Parallel workers: {workers}")

    # 加载 RAG 检索器
    print("[INFO] Loading code RAG retriever...")
    retriever = _load_code_rag_retriever()

    # 检查检索器是否可用
    if retriever.index['embeddings'] is None:
        print("[ERROR] Code RAG index not loaded. Please run code indexer first.")
        return

    # 覆盖模型路径为绝对路径
    retriever.model_name = os.path.join(_project_root, rag_embedding_model)

    print(f"[INFO] Code RAG index loaded: {len(retriever.index['chunks'])} chunks")
    print(f"[INFO] Using top_k={args.top_k}, max_chars={args.max_chars}")

    # 断点续传
    start_index = 0
    if args.start_from:
        if args.start_from in all_ops:
            start_index = all_ops.index(args.start_from)
            print(f"[INFO] Starting from op: {args.start_from} (index {start_index})")
        else:
            print(f"[WARN] Start op '{args.start_from}' not found, starting from beginning")

    for run in range(runs):
        out_dir = f"output/ascendc/rag_code_all/{temperature}-{top_p}/{model_name}/run{run}"
        os.makedirs(out_dir, exist_ok=True)

        ops_to_process = all_ops[start_index:]

        # 并行执行
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_op, op, dataset[op]['category'], out_dir, retriever, model
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