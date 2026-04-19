"""
KB query node for generator agent.

Query KB knowledge base (ChromaDB + LlamaIndex) for API documentation.
"""
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.kb_retriever import KBRetriever


def _ensure_english_for_kb(client, model: str, user_question: str) -> str:
    """Translate Chinese query to English for KB (KB requires English)."""
    prompt = (
        "Convert the following user text into a single English sentence suitable for "
        "knowledge-base search. Output only that English sentence, no quotes or explanation.\n\n"
        f"{user_question}"
    )
    content, _ = _call_llm(client, model, prompt)
    return content.strip().strip('"\'') or user_question


def _call_llm(client, model: str, prompt: str) -> tuple:
    """Simple LLM call helper."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    msg = resp.choices[0].message
    return (msg.content or "").strip(), ""


def kb_query_node(
    state: GeneratorAgentState,
    client,
    model: str,
    kb_retriever: KBRetriever = None,
) -> Dict[str, Any]:
    """
    KB query node: query knowledge base for API documentation.

    Args:
        state: Current agent state
        client: OpenAI client
        model: Model name
        kb_retriever: Optional pre-initialized KB retriever

    Returns:
        Dict with kb_results, query_round_count, tool_calls_log
    """
    # Initialize retriever if not provided
    if kb_retriever is None:
        kb_retriever = KBRetriever()

    # Get query
    query = (state.get("current_query") or "").strip()
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else ""

    # Fallback to user question if query is empty/short
    if not query or len(query) < 3:
        query = user_question.strip()

    # Ensure English for KB (KB works better with English queries)
    if query and not query.replace(" ", "").isascii():
        query = _ensure_english_for_kb(client, model, query)

    # Query KB
    chunks = []
    if kb_retriever.is_available():
        chunks = kb_retriever.retrieve(query, top_k=3)
    else:
        print("[WARN] KB not available, returning empty results")
        chunks = ["[KB unavailable: check chroma_db path configuration]"]

    # Update state
    round_num = state.get("query_round_count", 0) + 1
    response = "\n".join(chunks) if chunks else ""
    log_entry = {"round": round_num, "tool": "kb", "query": query, "response": response}

    print(f"[Round {round_num}] tool=kb query={query!r}")

    return {
        "kb_results": chunks,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }