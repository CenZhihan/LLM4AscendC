"""
KB (Knowledge Base) Retriever wrapper.

Reuses Agent_kernel's ChromaDB + LlamaIndex + BGE-M3 implementation.
"""
import os
import threading
from pathlib import Path
from typing import List, Tuple, Any, Optional

# Import config for consistent paths
from ...config import rag_embedding_model

# KB configuration
COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
# Default path: generator/agent/chroma_db (user confirmed to move here)
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(DEFAULT_PERSIST_DIR),
    )
).resolve()
# Use config.py setting for BGE-M3 model path
BGE_M3_PATH = Path(rag_embedding_model)

# Thread-safe singleton for ChromaDB client
_chroma_client: Any = None
_chroma_collection: Any = None
_chroma_lock = threading.Lock()


def _get_chroma_client_and_collection() -> Tuple[Any, Any]:
    """
    Get singleton ChromaDB client and collection (thread-safe).

    Returns:
        Tuple of (client, collection)
    """
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


class KBRetriever:
    """
    Wrapper for KB knowledge base retrieval (ChromaDB + LlamaIndex + BGE-M3).

    Provides a unified interface for querying API documentation from the KB.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        bge_m3_path: Optional[str] = None,
    ):
        """
        Initialize KB retriever.

        Args:
            persist_dir: Override default persist directory
            collection_name: Override default collection name
            bge_m3_path: Override default BGE-M3 model path
        """
        self.persist_dir = Path(persist_dir or PERSIST_DIR).resolve()
        self.collection_name = collection_name or COLLECTION_NAME
        self.bge_m3_path = Path(bge_m3_path or BGE_M3_PATH).resolve()
        self._client = None
        self._collection = None

    def _ensure_client(self):
        """Ensure ChromaDB client is initialized."""
        if self._client is None:
            self._client, self._collection = _get_chroma_client_and_collection()

    def is_available(self) -> bool:
        """Check if KB is available (has persisted data)."""
        self._ensure_client()
        if self._collection is None:
            return False
        count = self._collection.count()
        return count > 0

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant API documentation chunks from KB.

        Args:
            query: Query string (should be English for best results)
            top_k: Number of top results to retrieve

        Returns:
            List of formatted documentation chunks
        """
        self._ensure_client()

        from llama_index.core import VectorStoreIndex
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        chroma_client, chroma_collection = _get_chroma_client_and_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Load embedding model (BGE-M3)
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=str(self.bge_m3_path),
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARN] Failed to load BGE-M3 from {self.bge_m3_path}: {e}")
            # Fallback to default model
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-m3",
                trust_remote_code=True,
            )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        def _collect_seq_docs(source_file: str, seq_int: int) -> List[str]:
            """Collect all docs under same source_file and chunk_seq."""
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
            """Format docs with clear separation."""
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
            entry = "\n".join(entry_lines)
            out.append(entry)

        return out


def query_knowledge(question: str, top_k: int = 5) -> List[str]:
    """
    Convenience function for KB query (compatible with Agent_kernel interface).

    Args:
        question: Query question (English preferred)
        top_k: Number of top results

    Returns:
        List of retrieved documentation chunks
    """
    retriever = KBRetriever()
    return retriever.retrieve(question, top_k)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How can I use printf function"
    retriever = KBRetriever()
    if retriever.is_available():
        print(f"[INFO] KB available, querying: {q}")
        for t in retriever.retrieve(q, top_k=3):
            print(f"{t}\n")
    else:
        print(f"[WARN] KB not available at {PERSIST_DIR}")