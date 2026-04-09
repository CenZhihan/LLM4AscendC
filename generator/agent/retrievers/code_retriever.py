"""
Code RAG Retriever wrapper.

Wraps generator/rag/embedding_retriever.py for unified interface.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional

# Import from generator's existing RAG module
from generator.rag.embedding_retriever import EmbeddingRetriever
from generator.config import (
    rag_index_path,
    rag_embedding_model,
    rag_top_k,
    rag_max_chars,
)


class CodeRetriever:
    """
    Wrapper for Code RAG retrieval (EmbeddingRetriever + sentence-transformers).

    Provides unified interface for code snippet retrieval.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
        devices: Optional[List[str]] = None,
    ):
        """
        Initialize Code RAG retriever.

        Args:
            index_path: Override default index path
            model_name: Override default embedding model
            top_k: Override default top_k
            max_chars: Override default max chars for output
            devices: Device list (e.g., ['cpu'], ['npu:0'])
        """
        self.index_path = index_path or rag_index_path
        self.model_name = model_name or rag_embedding_model
        self.top_k = top_k or rag_top_k
        self.max_chars = max_chars or rag_max_chars
        self.devices = devices or ['cpu']  # Default to CPU to avoid NPU issues

        self._retriever: Optional[EmbeddingRetriever] = None
        self._loaded = False

    def _ensure_retriever(self) -> EmbeddingRetriever:
        """Ensure EmbeddingRetriever is initialized and loaded."""
        if self._retriever is None:
            self._retriever = EmbeddingRetriever(
                index_path=self.index_path,
                model_name=self.model_name,
                devices=self.devices,
            )
        if not self._loaded:
            self._loaded = self._retriever.load_index()
            if not self._loaded:
                print(f"[WARN] Failed to load Code RAG index from {self.index_path}")
        return self._retriever

    def is_available(self) -> bool:
        """Check if Code RAG is available (index loaded)."""
        self._ensure_retriever()
        return self._loaded and self._retriever.index['embeddings'] is not None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant code snippets.

        Args:
            query: Query string (e.g., "Ascend C GELU kernel implementation")
            top_k: Override default top_k

        Returns:
            List of formatted code snippets
        """
        retriever = self._ensure_retriever()
        if not self._loaded:
            return []

        k = top_k or self.top_k
        results = retriever.retrieve(query, top_k=k)

        # Format results
        formatted = []
        total_chars = 0

        for i, r in enumerate(results):
            code = r.get('code', '')
            file_path = r.get('file', 'unknown')
            score = r.get('score', 0.0)

            header = f"### Code Snippet {i+1} (score: {score:.3f}, file: {file_path})\n\n"
            code_block = f"```cpp\n{code}\n```\n\n"

            part = header + code_block
            if total_chars + len(part) > self.max_chars:
                remaining = self.max_chars - total_chars
                if remaining > 100:
                    formatted.append(part[:remaining] + "\n\n(... truncated ...)")
                break

            formatted.append(part)
            total_chars += len(part)

        return formatted

    def build_query(self, op_name: str, category: str, extra_context: Optional[str] = None) -> str:
        """
        Build an effective query for code RAG.

        Args:
            op_name: Operator name (e.g., "gelu")
            category: Operator category (e.g., "activation")
            extra_context: Extra context from LLM query

        Returns:
            Constructed query string
        """
        parts = [f"Ascend C {op_name} kernel implementation"]

        # Add category-specific keywords
        category_keywords = {
            "activation": ["Relu", "Gelu", "Sigmoid", "activation"],
            "matmul": ["MatMul", "Gemm", "matrix multiplication"],
            "convolution": ["Conv2D", "Convolution", "depthwise"],
            "attention": ["Attention", "Softmax", "scaled_dot_product"],
            "normalization": ["LayerNorm", "BatchNorm", "RmsNorm"],
            "pooling": ["MaxPool", "AvgPool", "Pool2D"],
            "loss": ["MSELoss", "CrossEntropy", "loss"],
            "reduce": ["ReduceSum", "ReduceMax", "reduction"],
            "broadcast": ["elementwise", "Add", "Mul"],
            "math": ["Exp", "Log", "Sqrt", "CumSum"],
            "index": ["Gather", "Scatter", "IndexSelect"],
        }

        if category in category_keywords:
            keywords = category_keywords[category][:2]
            parts.append(f"API: {' '.join(keywords)}")

        if extra_context:
            parts.append(extra_context)

        return '\n'.join(parts)


def query_code_rag(query: str, top_k: int = 3) -> List[str]:
    """
    Convenience function for Code RAG query.

    Args:
        query: Query string
        top_k: Number of results

    Returns:
        List of code snippets
    """
    retriever = CodeRetriever()
    return retriever.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Ascend C GELU kernel"
    retriever = CodeRetriever()
    if retriever.is_available():
        print(f"[INFO] Code RAG available, querying: {q}")
        for t in retriever.retrieve(q):
            print(f"{t}\n")
    else:
        print(f"[WARN] Code RAG not available (index at {rag_index_path})")