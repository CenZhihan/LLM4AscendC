"""
Environment check node for generator agent.

Check CANN environment compatibility and API availability.
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.env_checker import EnvCheckRetriever


def env_check_node(
    state: GeneratorAgentState,
    env_retriever: EnvCheckRetriever = None,
) -> Dict[str, Any]:
    """
    Environment check node: verify CANN environment and API compatibility.

    Args:
        state: Current agent state
        env_retriever: Optional pre-initialized EnvCheckRetriever

    Returns:
        Dict with env_check_results, query_round_count, tool_calls_log
    """
    # Initialize retriever if not provided
    if env_retriever is None:
        env_retriever = EnvCheckRetriever()

    # Get query from state or generate from context
    query = (state.get("current_query") or "").strip()
    if not query:
        # Build context-aware query from agent state
        op_name = state.get("op_name", "")
        category = state.get("category", "")
        if op_name:
            query = f"check environment for {op_name} operator"
        else:
            query = "check environment"

    # Perform environment check
    results = []
    if env_retriever.is_available():
        results = env_retriever.retrieve(query)
    else:
        print("[WARN] CANN environment not found, env check unavailable")
        results = ["[CANN 环境未找到，无法执行环境检查]"]

    # Update state
    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(results) if results else ""
    log_entry = {"round": round_num, "tool": "ENV_CHECK", "query": query, "response": response}

    print(f"[Round {round_num}] 工具=环境检查(ENV_CHECK), 查询=\"{query[:100]}\"")

    return {
        "env_check_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
