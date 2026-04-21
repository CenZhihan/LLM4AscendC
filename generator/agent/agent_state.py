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

from ..config import agent_max_query_rounds as MAX_QUERY_ROUNDS


def _add_list(left: List[str], right: List[str]) -> List[str]:
    """Reducer for list fields: append new results to existing."""
    return (left or []) + (right or [])


def _add_tool_calls(
    left: List[Dict[str, Any]], right: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reducer for tool call logs: append new logs."""
    return (left or []) + (right or [])


def _append_error_log(
    left: List[Dict[str, Any]], right: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reducer for tool-choice / parse error records."""
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
        next_action: Next tool action (lowercase tool key or ANSWER)
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
    env_check_results: Annotated[List[str], _add_list]
    kb_shell_search_results: Annotated[List[str], _add_list]
    api_lookup_results: Annotated[List[str], _add_list]
    api_constraint_results: Annotated[List[str], _add_list]
    api_alternative_results: Annotated[List[str], _add_list]
    tiling_calc_results: Annotated[List[str], _add_list]
    tiling_validate_results: Annotated[List[str], _add_list]
    npu_arch_results: Annotated[List[str], _add_list]
    code_style_results: Annotated[List[str], _add_list]
    security_check_results: Annotated[List[str], _add_list]
    ascend_search_results: Annotated[List[str], _add_list]
    ascend_fetch_results: Annotated[List[str], _add_list]
    ascend_search_allowed_urls: Annotated[List[str], _add_list]
    registered_tool_results: Annotated[List[str], _add_list]

    # Structured env check results (for programmatic access)
    env_check_env_result: NotRequired[Dict[str, Any]]    # Environment overview
    env_check_npu_result: NotRequired[Dict[str, Any]]    # NPU device query
    env_check_api_result: NotRequired[Dict[str, Any]]    # API compatibility check

    # Structured new tool results
    kb_shell_search_result: NotRequired[Dict[str, Any]]  # KB shell search
    api_lookup_result: NotRequired[Dict[str, Any]]       # API signature lookup
    api_constraint_result: NotRequired[Dict[str, Any]]   # API constraint check
    api_alternative_result: NotRequired[Dict[str, Any]]  # API alternative finder
    tiling_calc_result: NotRequired[Dict[str, Any]]      # Tiling calculation
    tiling_validate_result: NotRequired[Dict[str, Any]]  # Tiling validation
    npu_arch_result: NotRequired[Dict[str, Any]]         # NPU architecture query
    code_style_result: NotRequired[Dict[str, Any]]       # Code style check
    security_check_result: NotRequired[Dict[str, Any]]   # Security pattern check
    ascend_search_result: NotRequired[Dict[str, Any]]    # Ascend online docs search
    ascend_fetch_result: NotRequired[Dict[str, Any]]     # Ascend online docs fetch

    # Tool call logging
    tool_calls_log: Annotated[List[Dict[str, Any]], _add_tool_calls]

    # Control flow fields (NotRequired for optional presence)
    query_round_count: NotRequired[int]      # Current round number
    next_action: NotRequired[str]            # Next tool: KB, WEB, CODE_RAG, ANSWER
    current_query: NotRequired[str]          # Current query string
    tool_choice_json: NotRequired[Dict[str, Any]]  # Last parsed ToolChoiceV1 as dict (tool/query/args)
    reasoning_content: NotRequired[str]      # LLM reasoning output

    tool_choice_parse_failed: NotRequired[bool]
    tool_choice_error_log: Annotated[List[Dict[str, Any]], _append_error_log]

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
        "env_check_results": [],
        "kb_shell_search_results": [],
        "api_lookup_results": [],
        "api_constraint_results": [],
        "api_alternative_results": [],
        "tiling_calc_results": [],
        "tiling_validate_results": [],
        "npu_arch_results": [],
        "code_style_results": [],
        "security_check_results": [],
        "ascend_search_results": [],
        "ascend_fetch_results": [],
        "ascend_search_allowed_urls": [],
        "registered_tool_results": [],
        "tool_calls_log": [],
        "query_round_count": 0,
    }