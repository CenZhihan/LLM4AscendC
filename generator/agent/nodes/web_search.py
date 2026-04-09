"""
Web search node for generator agent.

Search web for relevant documentation and tutorials.
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.web_retriever import WebRetriever


def _generate_search_query(client, model: str, state: GeneratorAgentState) -> str:
    """Generate/refine search query based on context."""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_need = user_msgs[0].content if user_msgs else ""
    existing = state.get("web_results", [])
    existing_text = "\n".join(existing) if existing else ""

    if not existing_text:
        prompt = (
            f"用户需求描述：\n{user_need}\n\n"
            "请用一句话总结成一个适合在搜索引擎中查询的问题（只输出这句话，不要解释）。"
        )
    else:
        prompt = (
            f"用户需求：\n{user_need}\n\n已搜到的信息：\n{existing_text[:500]}\n\n"
            "请再提出一个不同的搜索问题来补充信息（一句话，只输出这句话）。"
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    query = (resp.choices[0].message.content or "").strip().strip('"\'')
    return query or user_need


def web_search_node(
    state: GeneratorAgentState,
    client,
    model: str,
    web_retriever: WebRetriever = None,
) -> Dict[str, Any]:
    """
    Web search node: search web for relevant content.

    Args:
        state: Current agent state
        client: OpenAI client
        model: Model name
        web_retriever: Optional pre-initialized Web retriever

    Returns:
        Dict with web_results, query_round_count, tool_calls_log
    """
    # Initialize retriever if not provided
    if web_retriever is None:
        web_retriever = WebRetriever()

    # Get or generate query
    query = (state.get("current_query") or "").strip()
    if not query:
        query = _generate_search_query(client, model, state)

    # Perform search
    results = []
    if web_retriever.is_available():
        results = web_retriever.retrieve(query)
    else:
        print("[WARN] Web search not available (pip install ddgs)")
        results = ["[网页搜索不可用，请安装 ddgs: pip install ddgs]"]

    # Update state
    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(results) if results else ""
    log_entry = {"round": round_num, "tool": "WEB", "query": query, "response": response}

    print(f"[Round {round_num}] 工具=网页搜索(WEB), 查询=\"{query}\"")

    return {
        "web_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }