"""
Chroma + llama-index knowledge retrieval for Agent KB tool.
Defaults: generation/agent/chroma_db and generation/agent/models/bge-m3.
Override with KB_PERSIST_DIR, KB_COLLECTION, BGE_M3_PATH via env (same as build_knowledge_base*.py).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Tuple

_AGENT_ROOT = Path(__file__).resolve().parent.parent

COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(_AGENT_ROOT / "chroma_db"),
    )
).resolve()
BGE_M3_PATH = Path(os.environ.get("BGE_M3_PATH", str(_AGENT_ROOT / "models" / "bge-m3"))).resolve()
EMBEDDING_DIM = 1024

_chroma_client: Any = None
_chroma_collection: Any = None
_chroma_lock = threading.Lock()


def _get_chroma_client_and_collection() -> Tuple[Any, Any]:
    global _chroma_client, _chroma_collection
    with _chroma_lock:
        if _chroma_client is None:
            import chromadb

            path = str(PERSIST_DIR)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=path)
            _chroma_collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    return _chroma_client, _chroma_collection


def query_knowledge(question: str, top_k: int = 5) -> List[str]:
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    chroma_client, chroma_collection = _get_chroma_client_and_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = HuggingFaceEmbedding(
        model_name=str(BGE_M3_PATH),
        trust_remote_code=True,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    def _collect_seq_docs(source_file: str, seq_int: int) -> List[str]:
        def _get_docs(val):
            where = {
                "$and": [
                    {"source_file": source_file},
                    {"chunk_seq": val},
                ]
            }
            raw = chroma_collection.get(where=where)
            return raw.get("documents") or []

        docs = _get_docs(seq_int)
        if not docs:
            docs = _get_docs(str(seq_int))
        return docs

    def _format_seq_docs(seq_int: int, docs: List[str]) -> str:
        if not docs:
            return ""
        parts: List[str] = []
        for idx, d in enumerate(docs, start=1):
            header = f"----- [seq={seq_int}, part={idx}] -----"
            parts.append(header)
            parts.append(d.strip())
        return "\n\n".join(parts)

    out: List[str] = []
    for rank, node in enumerate(nodes, start=1):
        meta = getattr(node, "metadata", None) or {}
        cur_seq_int = 0
        try:
            source_file = meta.get("source_file")
            chunk_seq = meta.get("chunk_seq")
            if source_file is not None and chunk_seq is not None:
                cur_seq_int = int(chunk_seq)
                cur_docs = _collect_seq_docs(str(source_file), cur_seq_int)
            else:
                cur_docs = []
        except Exception:
            cur_docs = []

        text = _format_seq_docs(cur_seq_int, cur_docs) if cur_docs else getattr(node, "text", None)
        if not text:
            continue

        entry_lines = [
            f"=== TOP {rank} ===",
            "[current]",
            text,
        ]
        out.append("\n".join(entry_lines))

    return out


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How can I use the assert function"
    for t in query_knowledge(q, top_k=3):
        print(f"{t}\n")
