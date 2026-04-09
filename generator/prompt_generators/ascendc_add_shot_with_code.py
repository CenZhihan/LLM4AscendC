from .prompt_registry import register_prompt, BasePromptStrategy
from .prompt_utils import read_relavant_files, ascendc_template
from ..rag import EmbeddingRetriever
from ..config import (
    rag_index_path, rag_embedding_model, rag_top_k, rag_max_chars
)


# 全局检索器实例（延迟初始化）
_retriever: EmbeddingRetriever = None


def _get_retriever() -> EmbeddingRetriever:
    """获取全局检索器实例"""
    global _retriever
    if _retriever is None:
        _retriever = EmbeddingRetriever(rag_index_path, rag_embedding_model)
        if not _retriever.load_index():
            print("[WARN] RAG index not found. Please run build_rag_index.py first.")
    return _retriever


def _build_query(op: str, arc_src: str) -> str:
    """
    构建检索查询

    Args:
        op: 操作名称
        arc_src: 架构源码

    Returns:
        查询字符串
    """
    # 使用操作名称和关键代码特征作为查询
    query_parts = [f"AscendC kernel implementation for {op}"]

    # 提取架构中的关键信息
    import re
    # 提取 forward 函数签名
    forward_match = re.search(r'def forward\([^)]*\)[^:]*:\s*([^\n]+)', arc_src)
    if forward_match:
        query_parts.append(forward_match.group(1).strip())

    # 提取类名
    class_match = re.search(r'class\s+(\w+)', arc_src)
    if class_match:
        query_parts.append(f"Model: {class_match.group(1)}")

    return '\n'.join(query_parts)


def _format_retrieved_code(results: list[dict], max_chars: int) -> str:
    """
    格式化检索结果

    Args:
        results: 检索结果列表
        max_chars: 最大字符数

    Returns:
        格式化后的字符串
    """
    if not results:
        return ""

    sections = []
    total_chars = 0

    for i, result in enumerate(results):
        code = result['code']
        file_path = result['file']
        score = result['score']

        # 提取文件名
        filename = file_path.split('ascendCode/')[-1] if 'ascendCode/' in file_path else file_path

        section = f"### 参考 {i+1}: {filename} (相似度: {score:.3f})\n```cpp\n{code}\n```\n"

        if total_chars + len(section) > max_chars:
            # 截断当前代码
            remaining = max_chars - total_chars - 100  # 预留格式化空间
            if remaining > 200:
                code = code[:remaining] + "\n// ... (已截断)"
                section = f"### 参考 {i+1}: {filename} (相似度: {score:.3f})\n```cpp\n{code}\n```\n"
                sections.append(section)
            break

        sections.append(section)
        total_chars += len(section)

    return '\n'.join(sections)


@register_prompt("ascendc", "add_shot_with_code")
class AscendcAddShotWithCodeStrategy(BasePromptStrategy):
    """基于 Embedding 检索的 AscendC prompt 策略"""

    def generate(self, op: str) -> str:
        """
        生成 prompt

        Args:
            op: 操作名称

        Returns:
            prompt 字符串
        """
        # 1. 生成基础 prompt
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')

        # 2. 检索相关代码
        retriever = _get_retriever()
        query = _build_query(op, arc_src)
        results = retriever.retrieve(query, top_k=rag_top_k)

        if not results:
            print(f"[INFO] No relevant code found for op: {op}")
            return base_prompt

        # 3. 格式化检索结果
        retrieved_section = _format_retrieved_code(results, rag_max_chars)

        # 4. 拼接到 prompt
        rag_header = "## 相似 AscendC 实现参考\n\n以下是从代码库中检索到的相似实现，可作为参考：\n\n"
        rag_footer = "\n---\n\n"

        return rag_header + retrieved_section + rag_footer + base_prompt