"""
Code RAG node for generator agent.

Retrieve relevant code snippets from the code library.
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.code_retriever import CodeRetriever


def code_rag_node(
    state: GeneratorAgentState,
    code_retriever: CodeRetriever = None,
) -> Dict[str, Any]:
    """
    Code RAG node: retrieve code snippets from code library.

    Args:
        state: Current agent state
        code_retriever: Optional pre-initialized Code retriever

    Returns:
        Dict with code_rag_results, query_round_count, tool_calls_log
    """
    # Initialize retriever if not provided
    if code_retriever is None:
        code_retriever = CodeRetriever()

    # Build query
    op_name = state.get("op_name", "")
    category = state.get("category", "")
    current_query = (state.get("current_query") or "").strip()

    # Use retriever's build_query method for better results
    query = code_retriever.build_query(op_name, category, current_query if current_query else None)

    # Retrieve code
    results = []
    if code_retriever.is_available():
        results = code_retriever.retrieve(query)
    else:
        print("[WARN] Code RAG not available (index not found)")
        results = [f"[代码检索不可用，索引路径: {code_retriever.index_path}]"]

    # Update state
    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(results) if results else ""
    log_entry = {"round": round_num, "tool": "code_rag", "query": query, "response": response}

    print(f"[Round {round_num}] 工具=代码检索(CODE_RAG), 查询=\"{query[:100]}...\"")

    return {
        "code_rag_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }