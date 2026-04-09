import os
import re
from typing import Generator
from .embedding_retriever import EmbeddingRetriever


# 代码片段的最小和最大字符数
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 6000


def extract_code_chunks(file_path: str) -> list[dict]:
    """
    从源文件中提取代码片段

    Args:
        file_path: 源文件路径

    Returns:
        代码片段列表，每个元素为 {'file': str, 'code': str, 'meta': dict}
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}")
        return []

    chunks = []
    lines = content.split('\n')

    # 方法1：按类/函数分割
    # 匹配 C++ 类定义、函数定义、extern "C" 函数
    pattern = r'((?:template\s*<[^>]*>\s*)?(?:class|struct)\s+\w+[^{]*\{)|(extern\s+"C"\s+__global__\s+__aicore__\s+void\s+\w+[^{]*\{)|((?:__aicore__\s+)?(?:inline\s+)?(?:void|int|auto)\s+\w+\s*\([^)]*\)\s*(?:const\s*)?\{)'

    matches = list(re.finditer(pattern, content))

    if matches:
        for i, match in enumerate(matches):
            start = match.start()
            # 找到结束位置（下一个匹配的开始或文件结束）
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            chunk_code = content[start:end].strip()

            # 提取名称（类名或函数名）
            name_match = re.search(r'(?:class|struct)\s+(\w+)|void\s+(\w+)\s*\(', match.group(0))
            name = name_match.group(1) or name_match.group(2) if name_match else "unknown"

            if MIN_CHUNK_CHARS <= len(chunk_code) <= MAX_CHUNK_CHARS:
                chunks.append({
                    'file': file_path,
                    'code': chunk_code,
                    'meta': {
                        'name': name,
                        'start_line': content[:start].count('\n') + 1,
                        'end_line': content[:end].count('\n') + 1
                    }
                })

    # 方法2：如果没有找到合适的分割，按固定行数分割
    if not chunks:
        chunk_lines = 50  # 每50行一个chunk
        for i in range(0, len(lines), chunk_lines):
            chunk_code = '\n'.join(lines[i:i + chunk_lines])
            if MIN_CHUNK_CHARS <= len(chunk_code) <= MAX_CHUNK_CHARS:
                chunks.append({
                    'file': file_path,
                    'code': chunk_code,
                    'meta': {
                        'name': os.path.basename(file_path),
                        'start_line': i + 1,
                        'end_line': min(i + chunk_lines, len(lines))
                    }
                })

    return chunks


def scan_code_files(code_dir: str, extensions: list[str] = ['.cpp', '.h']) -> Generator[str, None, None]:
    """
    扫描代码目录，返回所有匹配的文件路径

    Args:
        code_dir: 代码目录
        extensions: 文件扩展名列表

    Yields:
        文件路径
    """
    for root, dirs, files in os.walk(code_dir):
        # 跳过一些无关目录
        dirs[:] = [d for d in dirs if d not in ['build', 'build_out', '.git', '__pycache__', 'node_modules']]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                yield os.path.join(root, file)


class CodeIndexer:
    """代码索引器"""

    def __init__(self, code_dir: str = None, file_extensions: list[str] = None, retriever: EmbeddingRetriever = None):
        """
        初始化索引器

        Args:
            code_dir: 代码目录（可选，build_index 时指定）
            file_extensions: 文件扩展名列表（可选）
            retriever: EmbeddingRetriever 实例（可选，build_index 时创建）
        """
        self.code_dir = code_dir
        self.file_extensions = file_extensions or ['.cpp', '.h']
        self.retriever = retriever

    def collect_chunks(self, code_dir: str = None) -> list[dict]:
        """
        收集代码片段（不构建 embedding，只收集）

        Args:
            code_dir: 代码目录（可选，覆盖 self.code_dir）

        Returns:
            代码片段列表
        """
        code_dir = code_dir or self.code_dir
        if not code_dir:
            raise ValueError("code_dir must be specified")

        print(f"[INFO] Scanning {code_dir} for {self.file_extensions} files...")

        all_chunks = []
        file_count = 0

        for file_path in scan_code_files(code_dir, self.file_extensions):
            chunks = extract_code_chunks(file_path)
            all_chunks.extend(chunks)
            file_count += 1

            if file_count % 500 == 0:
                print(f"[INFO] Scanned {file_count} files, extracted {len(all_chunks)} chunks...")

        print(f"[INFO] Total: {file_count} files, {len(all_chunks)} chunks")
        return all_chunks

    def build_index(self, code_dir: str = None, extensions: list[str] = None) -> None:
        """
        构建代码索引

        Args:
            code_dir: 代码目录（可选，覆盖 self.code_dir）
            extensions: 文件扩展名列表（可选，覆盖 self.file_extensions）
        """
        code_dir = code_dir or self.code_dir
        extensions = extensions or self.file_extensions

        if not code_dir:
            raise ValueError("code_dir must be specified")

        all_chunks = self.collect_chunks(code_dir)

        # 构建索引
        if self.retriever:
            self.retriever.build_index(all_chunks)

    def save_index(self) -> None:
        """保存索引"""
        if self.retriever:
            self.retriever.save_index()

    def load_index(self) -> bool:
        """加载索引"""
        if self.retriever:
            return self.retriever.load_index()
        return False