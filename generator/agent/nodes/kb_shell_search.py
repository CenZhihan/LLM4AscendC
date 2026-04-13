"""
KB Shell Search node for generator agent.

Searches Knowledge/ directory documents using grep/find.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.kb_shell_search import KBShellSearchRetriever


def _extract_search_params(query: str) -> dict:
    """Extract search parameters from query."""
    params = {
        "category": "all",
        "operator_name": "",
        "query": query,
    }

    # Extract category
    category_map = {
        "api": "api", "tiling": "tiling", "arch": "arch",
        "architecture": "arch", "code-review": "code-review",
        "reduction": "reduction", "index": "index-tracking",
    }
    for keyword, cat in category_map.items():
        if keyword.lower() in query.lower():
            params["category"] = cat
            break

    # Extract operator name (common Ascend C operator names)
    op_match = re.search(r"(?:算子|operator|op)[:：]?\s*([A-Za-z_]\w+)", query, re.IGNORECASE)
    if op_match:
        params["operator_name"] = op_match.group(1)

    return params


def _format_for_display(result) -> str:
    """Format KBShellSearchResult for display."""
    if hasattr(result, "matches"):
        lines = [f"知识库搜索: category='{result.category}', operator='{result.operator_name}'"]
        lines.append(f"  匹配结果: {result.total_matches} 条")
        if result.matches:
            for i, m in enumerate(result.matches[:10], 1):
                file = m.get("file", "")
                line_num = m.get("line", "")
                content = m.get("content", "")
                lines.append(f"  {i}. {file}:{line_num}")
                lines.append(f"     {content}")
            if result.total_matches > 10:
                lines.append(f"  ... 共 {result.total_matches} 条匹配")
        else:
            lines.append("  未找到匹配结果。")
        if result.details:
            lines.append(f"  {result.details}")
        return "\n".join(lines)
    return str(result)


def kb_shell_search_node(
    state: GeneratorAgentState,
    kb_search_retriever: KBShellSearchRetriever = None,
) -> Dict[str, Any]:
    """
    KB Shell Search node.

    Args:
        state: Current agent state
        kb_search_retriever: Optional pre-initialized retriever

    Returns:
        Dict with kb_shell_search_results, kb_shell_search_result (structured),
        query_round_count, tool_calls_log
    """
    if kb_search_retriever is None:
        kb_search_retriever = KBShellSearchRetriever()

    if not kb_search_retriever.is_available():
        return {
            "kb_shell_search_results": ["[知识库目录未找到，无法执行搜索]"],
            "kb_shell_search_result": {"query": "", "category": "", "operator_name": "", "matches": [], "total_matches": 0},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    params = _extract_search_params(query)
    result = kb_search_retriever.search(
        category=params["category"],
        operator_name=params["operator_name"],
        query=params["query"],
    )

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "KB_SHELL_SEARCH", "query": query[:100], "response": display_text}

    print(f"[Round {round_num}] 工具=知识库搜索(KB_SHELL_SEARCH), category='{params['category']}', matches={result.total_matches}")

    return {
        "kb_shell_search_results": [display_text],
        "kb_shell_search_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
