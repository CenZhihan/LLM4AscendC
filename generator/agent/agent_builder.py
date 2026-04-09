"""
Agent builder: construct LangGraph StateGraph for kernel generation.

Builds a workflow with tool selection, retrieval nodes, and answer generation.
"""
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from openai import OpenAI

from .agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from .agent_config import AgentToolMode, get_llm_config_compatible
from .nodes import (
    choose_tool_node,
    kb_query_node,
    web_search_node,
    code_rag_node,
    answer_node,
)
from .retrievers import KBRetriever, WebRetriever, CodeRetriever


def _entry_node(state: GeneratorAgentState) -> dict:
    """Entry node: no-op, just pass through."""
    return {}


def _route_entry(tool_mode: AgentToolMode):
    """Create entry router function."""
    def router(state: GeneratorAgentState) -> str:
        if tool_mode == AgentToolMode.NO_TOOL:
            return "answer"
        return "choose_tool"
    return router


def _route_after_choose_tool(tool_mode: AgentToolMode):
    """Create router after choose_tool."""
    def router(state: GeneratorAgentState) -> str:
        action = (state.get("next_action") or "").upper()
        rounds = state.get("query_round_count", 0)

        # Must answer if at max rounds
        if rounds >= MAX_QUERY_ROUNDS:
            return "answer"

        # Route based on action
        if action == "ANSWER":
            return "answer"
        elif action == "KB" and tool_mode.has_kb():
            return "kb_query"
        elif action == "WEB" and tool_mode.has_web():
            return "web_search"
        elif action == "CODE_RAG" and tool_mode.has_code_rag():
            return "code_rag"
        else:
            return "answer"  # Fallback

    return router


def build_agent_app(
    tool_mode: AgentToolMode,
    llm_config: Optional[Dict[str, Any]] = None,
    kb_retriever: Optional[KBRetriever] = None,
    web_retriever: Optional[WebRetriever] = None,
    code_retriever: Optional[CodeRetriever] = None,
):
    """
    Build LangGraph StateGraph for kernel generation agent.

    Args:
        tool_mode: Tool mode specifying enabled retrieval tools
        llm_config: Optional LLM config (api_key, base_url, model)
        kb_retriever: Optional pre-initialized KB retriever
        web_retriever: Optional pre-initialized Web retriever
        code_retriever: Optional pre-initialized Code retriever

    Returns:
        Compiled StateGraph application
    """
    # Get LLM config
    config = llm_config or get_llm_config_compatible()
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    model = config["model"]

    # Initialize retrievers
    _kb_retriever = kb_retriever or (KBRetriever() if tool_mode.has_kb() else None)
    _web_retriever = web_retriever or (WebRetriever() if tool_mode.has_web() else None)
    _code_retriever = code_retriever or (CodeRetriever() if tool_mode.has_code_rag() else None)

    # Create node functions with closures
    def choose_tool_fn(state: GeneratorAgentState) -> dict:
        return choose_tool_node(state, client, model, tool_mode)

    def kb_query_fn(state: GeneratorAgentState) -> dict:
        return kb_query_node(state, client, model, _kb_retriever)

    def web_search_fn(state: GeneratorAgentState) -> dict:
        return web_search_node(state, client, model, _web_retriever)

    def code_rag_fn(state: GeneratorAgentState) -> dict:
        return code_rag_node(state, _code_retriever)

    def answer_fn(state: GeneratorAgentState) -> dict:
        return answer_node(state, client, model)

    # Build workflow
    workflow = StateGraph(GeneratorAgentState)

    # Add nodes
    workflow.add_node("entry", _entry_node)
    workflow.add_node("choose_tool", choose_tool_fn)

    if tool_mode.has_kb():
        workflow.add_node("kb_query", kb_query_fn)
    if tool_mode.has_web():
        workflow.add_node("web_search", web_search_fn)
    if tool_mode.has_code_rag():
        workflow.add_node("code_rag", code_rag_fn)

    workflow.add_node("answer", answer_fn)

    # Set entry point
    workflow.set_entry_point("entry")

    # Add conditional edges from entry
    workflow.add_conditional_edges(
        "entry",
        _route_entry(tool_mode),
        {"answer": "answer", "choose_tool": "choose_tool"}
    )

    # Add conditional edges from choose_tool
    route_after_choose = _route_after_choose_tool(tool_mode)
    conditional_map = {"answer": "answer"}
    if tool_mode.has_kb():
        conditional_map["kb_query"] = "kb_query"
    if tool_mode.has_web():
        conditional_map["web_search"] = "web_search"
    if tool_mode.has_code_rag():
        conditional_map["code_rag"] = "code_rag"

    workflow.add_conditional_edges(
        "choose_tool",
        route_after_choose,
        conditional_map
    )

    # Add edges back to choose_tool from retrieval nodes
    if tool_mode.has_kb():
        workflow.add_edge("kb_query", "choose_tool")
    if tool_mode.has_web():
        workflow.add_edge("web_search", "choose_tool")
    if tool_mode.has_code_rag():
        workflow.add_edge("code_rag", "choose_tool")

    # Answer leads to END
    workflow.add_edge("answer", END)

    # Compile
    return workflow.compile()


# Convenience function for quick app creation
def create_agent(
    tool_mode: str = "no_tool",
    **kwargs,
):
    """
    Create agent app with string tool mode.

    Args:
        tool_mode: Tool mode string (no_tool, kb_only, web_only, etc.)
        **kwargs: Additional args passed to build_agent_app

    Returns:
        Compiled StateGraph application
    """
    mode = AgentToolMode(tool_mode)
    return build_agent_app(mode, **kwargs)