"""
Agent state definition for LangGraph-based kernel generation agent.

Defines GeneratorAgentState that integrates KB, WEB, and Code RAG retrieval results.
"""
from typing import Annotated, List, Dict, Any

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # Python < 3.11

from langgraph.graph import MessagesState


# Maximum query rounds before forced ANSWER
MAX_QUERY_ROUNDS = 3


def _add_list(left: List[str], right: List[str]) -> List[str]:
    """Reducer for list fields: append new results to existing."""
    return (left or []) + (right or [])


def _add_tool_calls(
    left: List[Dict[str, Any]], right: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reducer for tool call logs: append new logs."""
    return (left or []) + (right or [])


class GeneratorAgentState(MessagesState):
    """
    Unified state for generator agent with KB, WEB, and Code RAG retrieval.

    Inherits MessagesState for conversation history support.

    Retrieval results are annotated with reducers for multi-round accumulation.
    Control fields use NotRequired for optional presence.

    Attributes:
        kb_results: KB knowledge base results (API documentation)
        web_results: Web search results
        code_rag_results: Code RAG retrieval results
        tool_calls_log: Tool call history for debugging/reporting
        query_round_count: Current round number (0 to MAX_QUERY_ROUNDS)
        next_action: Next tool action (KB, WEB, CODE_RAG, ANSWER)
        current_query: Current query string for this round
        reasoning_content: LLM reasoning output (if supported)
        op_name: Operator name (e.g., "gelu")
        category: Operator category (e.g., "activation")
        language: Target language (e.g., "ascendc")
        strategy_name: Prompt strategy name (e.g., "add_shot")
        base_prompt: Base prompt from prompt_generators
    """

    # Retrieval results (annotated with reducers for multi-round accumulation)
    kb_results: Annotated[List[str], _add_list]
    web_results: Annotated[List[str], _add_list]
    code_rag_results: Annotated[List[str], _add_list]

    # Tool call logging
    tool_calls_log: Annotated[List[Dict[str, Any]], _add_tool_calls]

    # Control flow fields (NotRequired for optional presence)
    query_round_count: NotRequired[int]      # Current round number
    next_action: NotRequired[str]            # Next tool: KB, WEB, CODE_RAG, ANSWER
    current_query: NotRequired[str]          # Current query string
    reasoning_content: NotRequired[str]      # LLM reasoning output

    # Task context fields
    op_name: NotRequired[str]                # Operator name (e.g., "gelu")
    category: NotRequired[str]               # Operator category (e.g., "activation")
    language: NotRequired[str]               # Target language (e.g., "ascendc")
    strategy_name: NotRequired[str]          # Prompt strategy name (e.g., "add_shot")
    base_prompt: NotRequired[str]            # Base prompt from prompt_generators


def create_initial_state(
    base_prompt: str,
    op_name: str,
    category: str,
    language: str = "ascendc",
    strategy_name: str = "add_shot",
) -> Dict[str, Any]:
    """
    Create initial state for agent invocation.

    Args:
        base_prompt: The prompt from prompt_generators
        op_name: Operator name
        category: Operator category
        language: Target language (default: "ascendc")
        strategy_name: Prompt strategy name (default: "add_shot")

    Returns:
        Initial state dict for app.invoke()
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=base_prompt)],
        "op_name": op_name,
        "category": category,
        "language": language,
        "strategy_name": strategy_name,
        "base_prompt": base_prompt,
        "kb_results": [],
        "web_results": [],
        "code_rag_results": [],
        "tool_calls_log": [],
        "query_round_count": 0,
    }