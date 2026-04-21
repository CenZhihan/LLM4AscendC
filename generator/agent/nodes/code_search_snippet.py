"""Restricted snippet search node for curated local Ascend C sources."""
from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..retrievers.code_search_snippet_retriever import CodeSearchSnippetRetriever


def code_search_snippet_node(
    state: GeneratorAgentState,
    retriever: CodeSearchSnippetRetriever = None,
) -> Dict[str, Any]:
    if retriever is None:
        retriever = CodeSearchSnippetRetriever()

    op_name = state.get("op_name", "")
    category = state.get("category", "")
    current_query = (state.get("current_query") or "").strip()
    tool_choice = state.get("tool_choice_json") or {}
    args = tool_choice.get("args") or {}
    source = str(args.get("source") or "all")

    query = retriever.build_query(op_name, category, current_query if current_query else None)

    if retriever.is_available():
        results = retriever.retrieve(query=query, source=source)
    else:
        sources = retriever.available_sources()
        results = [
            "[code_search_snippet] unavailable. "
            f"cann_skills={sources['cann_skills'] or 'missing'}, "
            f"asc_devkit={sources['asc_devkit'] or 'missing'}"
        ]

    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(results) if results else ""
    log_entry = {
        "round": round_num,
        "tool": "code_search_snippet",
        "query": query,
        "response": response,
    }

    print(f"[Round {round_num}] 工具=片段检索(CODE_SEARCH_SNIPPECT), 查询=\"{query[:100]}...\"")

    return {
        "code_search_snippet_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }