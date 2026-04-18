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
            f"User goal:\n{user_need}\n\n"
            "Summarize in one short line as a web search query. Output only that line, no explanation."
        )
    else:
        prompt = (
            f"User goal:\n{user_need}\n\nExisting search snippets:\n{existing_text[:500]}\n\n"
            "Propose one different search query to fill gaps. Output only that line, no explanation."
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
        results = ["[Web search unavailable: pip install ddgs]"]

    # Update state
    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(results) if results else ""
    log_entry = {"round": round_num, "tool": "web", "query": query, "response": response}

    print(f"[Round {round_num}] tool=web query={query!r}")

    return {
        "web_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }