"""
Unified operator generation entry point.

Supports multiple generation strategies:
- rag: RAG-enhanced generation with code retrieval
- add_shot: Basic few-shot prompting
- selected_shot: Category-specific few-shot

Usage:
    python3 tools/generate_operator.py --model gpt-4 --strategy rag --categories activation
    python3 tools/generate_operator.py --model deepseek-chat --categories all --workers 4
    python3 tools/generate_operator.py --build-index --code-dir ascendCode
"""
import os
import sys
import argparse

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import generator module
from generator import (
    dataset, category2exampleop,
    rag_index_path, rag_embedding_model, rag_top_k, rag_max_chars,
    rag_code_dir, rag_file_extensions, ref_impl_base_path,
    temperature, top_p
)
from generator.rag import EmbeddingRetriever, CodeIndexer


def build_rag_index(code_dir: str, model_path: str = None):
    """Build RAG index from code directory."""
    if model_path is None:
        model_path = rag_embedding_model

    print(f"[INFO] Building RAG index from: {code_dir}")
    print(f"[INFO] Using embedding model: {model_path}")
    print(f"[INFO] Index will be saved to: {rag_index_path}")

    # Check code directory
    if not os.path.exists(code_dir):
        print(f"[ERROR] Code directory not found: {code_dir}")
        return False

    # Indexer
    indexer = CodeIndexer(
        code_dir=code_dir,
        file_extensions=rag_file_extensions
    )

    # Collect code chunks
    chunks = indexer.collect_chunks()
    if not chunks:
        print(f"[ERROR] No code files found in {code_dir}")
        return False

    print(f"[INFO] Collected {len(chunks)} code chunks")

    # Retriever
    retriever = EmbeddingRetriever(
        index_path=rag_index_path,
        model_name=model_path,
        devices=['cpu']  # Use CPU for indexing
    )

    # Build and save index
    retriever.build_index(chunks, batch_size=8)
    retriever.save_index()

    print(f"[SUCCESS] Index built with {len(chunks)} chunks")
    return True


def generate_with_rag(model: str, categories: list, workers: int, output_dir: str,
                      start_from: str = None, top_k: int = None, max_chars: int = None):
    """Generate operators using RAG-enhanced prompts."""
    from generator.scripts.generation.generate_rag import (
        _load_code_rag_retriever, _generate_one_op, generate_prompt_rag_code
    )
    from generator.utils import get_client
    from concurrent.futures import ThreadPoolExecutor, as_completed

    top_k = top_k or rag_top_k
    max_chars = max_chars or rag_max_chars

    # Get operator list
    all_ops = list(dataset.keys())
    if categories != ['all']:
        all_ops = [op for op in all_ops if dataset[op]['category'] in categories]
    all_ops = sorted(all_ops)

    print(f"[INFO] Total ops to generate: {len(all_ops)}")
    print(f"[INFO] Categories: {categories}")
    print(f"[INFO] Parallel workers: {workers}")
    print(f"[INFO] Output directory: {output_dir}")

    # Load RAG retriever
    print("[INFO] Loading code RAG retriever...")
    retriever = _load_code_rag_retriever()

    if retriever.index['embeddings'] is None:
        print("[ERROR] RAG index not loaded. Run with --build-index first.")
        return False

    print(f"[INFO] RAG index loaded: {len(retriever.index['chunks'])} chunks")

    # Resume support
    start_index = 0
    if start_from:
        if start_from in all_ops:
            start_index = all_ops.index(start_from)
            print(f"[INFO] Resuming from: {start_from} (index {start_index})")

    os.makedirs(output_dir, exist_ok=True)
    ops_to_process = all_ops[start_index:]

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_one_op, op, dataset[op]['category'], output_dir, retriever, model
            ): op
            for op in ops_to_process
        }

        for future in as_completed(futures):
            op, err = future.result()
            if err is None:
                success_count += 1
            else:
                fail_count += 1

    print(f"\n[SUMMARY] Generated {success_count} operators, {fail_count} failed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate AscendC operator kernels using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build RAG index from code library
  python3 tools/generate_operator.py --build-index --code-dir ascendCode

  # Generate activation operators with RAG
  python3 tools/generate_operator.py --model gpt-4 --strategy rag --categories activation

  # Generate all operators with 4 workers
  python3 tools/generate_operator.py --model deepseek-chat --categories all --workers 4

  # Resume from specific operator
  python3 tools/generate_operator.py --model gpt-4 --start-from softmax
"""
    )

    # Generation options
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="LLM model name (default: deepseek-chat)")
    parser.add_argument("--strategy", type=str, default="rag",
                        choices=["rag", "add_shot", "selected_shot"],
                        help="Generation strategy (default: rag)")
    parser.add_argument("--categories", nargs='+', default=['activation'],
                        help="Operator categories (default: activation). Use 'all' for all.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/<strategy>/<model>)")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume from specific operator")
    parser.add_argument("--top-k", type=int, default=rag_top_k,
                        help=f"RAG top-k (default: {rag_top_k})")
    parser.add_argument("--max-chars", type=int, default=rag_max_chars,
                        help=f"RAG max chars (default: {rag_max_chars})")

    # RAG index options
    parser.add_argument("--build-index", action="store_true",
                        help="Build RAG index before generation")
    parser.add_argument("--code-dir", type=str, default=rag_code_dir,
                        help=f"Code directory for RAG indexing (default: {rag_code_dir})")
    parser.add_argument("--embedding-model", type=str, default=rag_embedding_model,
                        help=f"Embedding model path (default: {rag_embedding_model})")

    args = parser.parse_args()

    # Determine output directory
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    output_dir = args.output_dir or f"output/{args.strategy}/{temperature}-{top_p}/{model_name}"

    # Build index if requested
    if args.build_index:
        success = build_rag_index(args.code_dir, args.embedding_model)
        if not success:
            sys.exit(1)
        print("[INFO] Index built successfully. You can now run generation.")
        return

    # Generate based on strategy
    if args.strategy == "rag":
        success = generate_with_rag(
            model=args.model,
            categories=args.categories,
            workers=args.workers,
            output_dir=output_dir,
            start_from=args.start_from,
            top_k=args.top_k,
            max_chars=args.max_chars
        )
    elif args.strategy == "add_shot":
        # TODO: Implement add_shot strategy
        print("[WARN] add_shot strategy not yet implemented in unified entry")
        success = False
    elif args.strategy == "selected_shot":
        # TODO: Implement selected_shot strategy
        print("[WARN] selected_shot strategy not yet implemented in unified entry")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()