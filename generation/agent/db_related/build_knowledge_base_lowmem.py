"""
低内存版本的知识库建库脚本（LLM4AscendC / generation/agent）。

区别于 build_knowledge_base.py：
- 只使用 pypdf 按行 + 正则做章节切分；
- 读取 PDF 时以 KB_PAGES_PER_BLOCK 页为一组处理；
- 持久化目录默认 generation/agent/chroma_db；嵌入模型默认 generation/agent/models/bge-m3
  （可用 BGE_M3_PATH、KB_PERSIST_DIR、KB_KNOWLEDGE_DIR 覆盖）。
"""

import os
import shutil
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent.parent

COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(_AGENT_ROOT / "chroma_db"),
    )
).resolve()
KNOWLEDGE_DIR = Path(
    os.environ.get("KB_KNOWLEDGE_DIR", str(_AGENT_ROOT / "Knowledge"))
).resolve()
BGE_M3_PATH = Path(
    os.environ.get("BGE_M3_PATH", str(_AGENT_ROOT / "models" / "bge-m3"))
).resolve()
EMBEDDING_DIM = 1024

REBUILD = os.environ.get("KB_REBUILD", "1") in ("1", "true", "True", "yes", "YES")

SKIP_FIRST_PAGES = int(os.environ.get("KB_SKIP_FIRST_PAGES", "22"))


def _iter_pdf_sections_pypdf_10pages(pdf_path: Path, pages_per_block: int = 10):
    from llama_index.core import Document
    from pypdf import PdfReader
    import re

    header_re = re.compile(r"^(?:\d+(?:\.\d+)*\s+.+|Chapter\s+\d+.*|第[一二三四五六七八九十\d]+章.*)$")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    current_lines = []
    saw_any_text = False
    chunk_seq = 0
    source_file = pdf_path.name

    for start_page in range(1, total_pages + 1, pages_per_block):
        end_page = min(start_page + pages_per_block - 1, total_pages)
        print(f"    [pypdf-10pages] pages {start_page}-{end_page}/{total_pages}")

        for page_idx in range(start_page, end_page + 1):
            page = reader.pages[page_idx - 1]
            if page_idx <= SKIP_FIRST_PAGES:
                continue
            text = page.extract_text() or ""
            if text.strip():
                saw_any_text = True

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if header_re.match(line):
                    if current_lines:
                        yield Document(
                            text="\n".join(current_lines).strip(),
                            metadata={"source_file": source_file, "chunk_seq": chunk_seq},
                        )
                        chunk_seq += 1
                        current_lines = []
                    current_lines.append(line)
                else:
                    current_lines.append(line)

    if current_lines:
        yield Document(
            text="\n".join(current_lines).strip(),
            metadata={"source_file": source_file, "chunk_seq": chunk_seq},
        )
    elif not saw_any_text:
        yield Document(text="(未提取到文本)", metadata={"source_file": source_file, "chunk_seq": chunk_seq})


def _run_build_lowmem() -> None:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import chromadb

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"知识库目录不存在: {KNOWLEDGE_DIR}")
    if not BGE_M3_PATH.exists():
        raise FileNotFoundError(
            f"本地 BGE-M3 不存在: {BGE_M3_PATH}，请下载到该路径或设置 BGE_M3_PATH"
        )

    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"未在 {KNOWLEDGE_DIR} 下找到 PDF 文件")

    embed_model = HuggingFaceEmbedding(
        model_name=str(BGE_M3_PATH),
        trust_remote_code=True,
    )

    if REBUILD and PERSIST_DIR.exists():
        print(f"[rebuild] 删除持久化目录: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    batch_size = int(os.environ.get("KB_INSERT_BATCH_SIZE", "2"))
    pages_per_block = int(os.environ.get("KB_PAGES_PER_BLOCK", "10"))
    total_inserted = 0

    for pdf_path in pdf_files:
        print(f"正在按章节解析 PDF（低内存模式，每次 {pages_per_block} 页）: {pdf_path.name} ...")
        print(f"  开始流式写入（batch_size={batch_size}），进度将显示已写入章节数...")
        batch = []
        for doc in _iter_pdf_sections_pypdf_10pages(pdf_path, pages_per_block=pages_per_block):
            batch.append(doc)
            if len(batch) >= batch_size:
                if hasattr(index, "insert_documents"):
                    index.insert_documents(batch)  # type: ignore[attr-defined]
                    total_inserted += len(batch)
                else:
                    for d in batch:
                        index.insert(d)  # type: ignore[attr-defined]
                        total_inserted += 1
                if total_inserted % 50 == 0:
                    print(f"    已写入章节块: {total_inserted}")
                batch.clear()

        if batch:
            if hasattr(index, "insert_documents"):
                index.insert_documents(batch)  # type: ignore[attr-defined]
                total_inserted += len(batch)
            else:
                for d in batch:
                    index.insert(d)  # type: ignore[attr-defined]
                    total_inserted += 1
            batch.clear()
        print(f"  完成该 PDF，累计写入章节块: {total_inserted}")

    if total_inserted <= 0:
        raise RuntimeError("未写入任何章节内容，请检查 PDF 解析结果。")

    print(f"（低内存模式）知识库构建完成，共写入 {total_inserted} 个章节块。")


def main() -> None:
    _run_build_lowmem()


if __name__ == "__main__":
    main()
