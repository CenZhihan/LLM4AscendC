"""
Agent builder: construct LangGraph StateGraph for kernel generation.

Builds a workflow with tool selection, retrieval nodes, and answer generation.
"""
from typing import Dict, Any, Optional, Union

from langgraph.graph import StateGraph, END
from openai import OpenAI

from .agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from .agent_config import (
    AgentToolMode,
    ToolType,
    NO_TOOL,
    has_kb,
    has_web,
    has_code_rag,
    has_env_check_env,
    has_env_check_npu,
    has_env_check_api,
    parse_tool_mode,
    get_llm_config_compatible,
)
from .nodes import (
    choose_tool_node,
    kb_query_node,
    web_search_node,
    code_rag_node,
    env_check_env_node,
    env_check_npu_node,
    env_check_api_node,
    answer_node,
)
from .retrievers import KBRetriever, WebRetriever, CodeRetriever
from .retrievers.env_checker import EnvCheckRetriever


def _entry_node(state: GeneratorAgentState) -> dict:
    """Entry node: no-op, just pass through."""
    return {}


def _route_entry(tool_mode: AgentToolMode):
    """Create entry router function."""
    def router(state: GeneratorAgentState) -> str:
        if tool_mode == NO_TOOL:
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
        elif action == "KB" and has_kb(tool_mode):
            return "kb_query"
        elif action == "WEB" and has_web(tool_mode):
            return "web_search"
        elif action == "CODE_RAG" and has_code_rag(tool_mode):
            return "code_rag"
        elif action == "ENV_CHECK_ENV" and has_env_check_env(tool_mode):
            return "env_check_env"
        elif action == "ENV_CHECK_NPU" and has_env_check_npu(tool_mode):
            return "env_check_npu"
        elif action == "ENV_CHECK_API" and has_env_check_api(tool_mode):
            return "env_check_api"
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
        tool_mode: Tool mode (FrozenSet[ToolType]) specifying enabled retrieval tools
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

    # Initialize retrievers (only if tool is enabled)
    _kb_retriever = kb_retriever or (KBRetriever() if has_kb(tool_mode) else None)
    _web_retriever = web_retriever or (WebRetriever() if has_web(tool_mode) else None)
    _code_retriever = code_retriever or (CodeRetriever() if has_code_rag(tool_mode) else None)
    _env_retriever = EnvCheckRetriever() if (
        has_env_check_env(tool_mode) or has_env_check_npu(tool_mode) or has_env_check_api(tool_mode)
    ) else None

    # Create node functions with closures
    def choose_tool_fn(state: GeneratorAgentState) -> dict:
        return choose_tool_node(state, client, model, tool_mode)

    def kb_query_fn(state: GeneratorAgentState) -> dict:
        return kb_query_node(state, client, model, _kb_retriever)

    def web_search_fn(state: GeneratorAgentState) -> dict:
        return web_search_node(state, client, model, _web_retriever)

    def code_rag_fn(state: GeneratorAgentState) -> dict:
        return code_rag_node(state, _code_retriever)

    def env_check_env_fn(state: GeneratorAgentState) -> dict:
        return env_check_env_node(state, _env_retriever)

    def env_check_npu_fn(state: GeneratorAgentState) -> dict:
        return env_check_npu_node(state, _env_retriever)

    def env_check_api_fn(state: GeneratorAgentState) -> dict:
        return env_check_api_node(state, _env_retriever)

    def answer_fn(state: GeneratorAgentState) -> dict:
        return answer_node(state, client, model)

    # Build workflow
    workflow = StateGraph(GeneratorAgentState)

    # Add nodes
    workflow.add_node("entry", _entry_node)
    workflow.add_node("choose_tool", choose_tool_fn)

    # Add retrieval nodes only if corresponding tool is enabled
    if has_kb(tool_mode):
        workflow.add_node("kb_query", kb_query_fn)
    if has_web(tool_mode):
        workflow.add_node("web_search", web_search_fn)
    if has_code_rag(tool_mode):
        workflow.add_node("code_rag", code_rag_fn)
    if has_env_check_env(tool_mode):
        workflow.add_node("env_check_env", env_check_env_fn)
    if has_env_check_npu(tool_mode):
        workflow.add_node("env_check_npu", env_check_npu_fn)
    if has_env_check_api(tool_mode):
        workflow.add_node("env_check_api", env_check_api_fn)

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
    if has_kb(tool_mode):
        conditional_map["kb_query"] = "kb_query"
    if has_web(tool_mode):
        conditional_map["web_search"] = "web_search"
    if has_code_rag(tool_mode):
        conditional_map["code_rag"] = "code_rag"
    if has_env_check_env(tool_mode):
        conditional_map["env_check_env"] = "env_check_env"
    if has_env_check_npu(tool_mode):
        conditional_map["env_check_npu"] = "env_check_npu"
    if has_env_check_api(tool_mode):
        conditional_map["env_check_api"] = "env_check_api"

    workflow.add_conditional_edges(
        "choose_tool",
        route_after_choose,
        conditional_map
    )

    # Add edges back to choose_tool from retrieval nodes
    if has_kb(tool_mode):
        workflow.add_edge("kb_query", "choose_tool")
    if has_web(tool_mode):
        workflow.add_edge("web_search", "choose_tool")
    if has_code_rag(tool_mode):
        workflow.add_edge("code_rag", "choose_tool")
    if has_env_check_env(tool_mode):
        workflow.add_edge("env_check_env", "choose_tool")
    if has_env_check_npu(tool_mode):
        workflow.add_edge("env_check_npu", "choose_tool")
    if has_env_check_api(tool_mode):
        workflow.add_edge("env_check_api", "choose_tool")

    # Answer leads to END
    workflow.add_edge("answer", END)

    # Compile
    return workflow.compile()


def create_agent(
    tool_mode: Union[str, AgentToolMode] = "no_tool",
    **kwargs,
):
    """
    Create agent app with flexible tool mode input.

    Args:
        tool_mode: Tool mode specification, can be:
            - str: "no_tool", "kb_only", "kb,web", "all", etc.
            - AgentToolMode: FrozenSet[ToolType]
        **kwargs: Additional args passed to build_agent_app

    Returns:
        Compiled StateGraph application

    Examples:
        create_agent("no_tool")
        create_agent("kb_only")
        create_agent("kb,web")
        create_agent("all")
        create_agent(frozenset({ToolType.KB, ToolType.WEB}))
    """
    mode = parse_tool_mode(tool_mode)
    return build_agent_app(mode, **kwargs)