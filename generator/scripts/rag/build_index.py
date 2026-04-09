# 构建 RAG 代码索引
import os
import sys
# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
from generator.rag import EmbeddingRetriever, CodeIndexer
from generator.config import rag_index_path, rag_embedding_model, rag_code_dir, rag_file_extensions


def main():
    parser = argparse.ArgumentParser(description="构建代码 RAG 索引")
    parser.add_argument("--code-dir", type=str, default=rag_code_dir, help="代码库目录")
    parser.add_argument("--index-path", type=str, default=rag_index_path, help="索引保存路径")
    parser.add_argument("--model", type=str, default=rag_embedding_model, help="Embedding 模型路径")
    parser.add_argument("--extensions", nargs='+', default=rag_file_extensions, help="文件扩展名")
    parser.add_argument("--batch-size", type=int, default=8, help="编码批大小")
    args = parser.parse_args()

    print(f"[INFO] Code directory: {args.code_dir}")
    print(f"[INFO] Index path: {args.index_path}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Extensions: {args.extensions}")

    # 检查代码目录是否存在
    if not os.path.exists(args.code_dir):
        print(f"[ERROR] Code directory not found: {args.code_dir}")
        print("[INFO] Please prepare your AscendC code library first.")
        return

    # 创建检索器
    retriever = EmbeddingRetriever(
        index_path=args.index_path,
        model_name=args.model,
        devices=['cpu']  # 使用 CPU 避免内存问题
    )

    # 创建索引器
    indexer = CodeIndexer(retriever)

    # 构建索引
    print("[INFO] Building index...")
    indexer.build_index(args.code_dir, args.extensions)

    # 保存索引
    print("[INFO] Saving index...")
    indexer.save_index()

    print("[INFO] Index building completed.")


if __name__ == "__main__":
    main()