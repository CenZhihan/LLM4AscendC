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
    has_kb_shell_search,
    has_api_lookup,
    has_api_constraint,
    has_api_alternative,
    has_tiling_calc,
    has_tiling_validate,
    has_npu_arch,
    has_code_style,
    has_security_check,
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
    npu_arch_node,
    tiling_calc_node,
    tiling_validate_node,
    api_lookup_node,
    api_constraint_node,
    api_alternative_node,
    code_style_node,
    security_check_node,
    kb_shell_search_node,
    answer_node,
)
from .retrievers import KBRetriever, WebRetriever, CodeRetriever
from .retrievers.env_checker import EnvCheckRetriever
from .retrievers.npu_arch_retriever import NpuArchRetriever
from .retrievers.tiling_retriever import TilingRetriever
from .retrievers.api_doc_retriever import ApiDocRetriever
from .retrievers.code_quality_retriever import CodeQualityRetriever
from .retrievers.kb_shell_search import KBShellSearchRetriever


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
        elif action == "KB_SHELL_SEARCH" and has_kb_shell_search(tool_mode):
            return "kb_shell_search"
        elif action == "API_LOOKUP" and has_api_lookup(tool_mode):
            return "api_lookup"
        elif action == "API_CONSTRAINT" and has_api_constraint(tool_mode):
            return "api_constraint"
        elif action == "API_ALTERNATIVE" and has_api_alternative(tool_mode):
            return "api_alternative"
        elif action == "TILING_CALC" and has_tiling_calc(tool_mode):
            return "tiling_calc"
        elif action == "TILING_VALIDATE" and has_tiling_validate(tool_mode):
            return "tiling_validate"
        elif action == "NPU_ARCH" and has_npu_arch(tool_mode):
            return "npu_arch"
        elif action == "CODE_STYLE" and has_code_style(tool_mode):
            return "code_style"
        elif action == "SECURITY_CHECK" and has_security_check(tool_mode):
            return "security_check"
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
    _npu_arch_retriever = NpuArchRetriever() if has_npu_arch(tool_mode) else None
    _tiling_retriever = TilingRetriever() if (
        has_tiling_calc(tool_mode) or has_tiling_validate(tool_mode)
    ) else None
    _api_retriever = ApiDocRetriever() if (
        has_api_lookup(tool_mode) or has_api_constraint(tool_mode) or has_api_alternative(tool_mode)
    ) else None
    _code_quality_retriever = CodeQualityRetriever() if (
        has_code_style(tool_mode) or has_security_check(tool_mode)
    ) else None
    _kb_shell_retriever = KBShellSearchRetriever() if has_kb_shell_search(tool_mode) else None

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

    def npu_arch_fn(state: GeneratorAgentState) -> dict:
        return npu_arch_node(state, _npu_arch_retriever)

    def tiling_calc_fn(state: GeneratorAgentState) -> dict:
        return tiling_calc_node(state, _tiling_retriever)

    def tiling_validate_fn(state: GeneratorAgentState) -> dict:
        return tiling_validate_node(state, _tiling_retriever)

    def api_lookup_fn(state: GeneratorAgentState) -> dict:
        return api_lookup_node(state, _api_retriever)

    def api_constraint_fn(state: GeneratorAgentState) -> dict:
        return api_constraint_node(state, _api_retriever)

    def api_alternative_fn(state: GeneratorAgentState) -> dict:
        return api_alternative_node(state, _api_retriever)

    def code_style_fn(state: GeneratorAgentState) -> dict:
        return code_style_node(state, _code_quality_retriever)

    def security_check_fn(state: GeneratorAgentState) -> dict:
        return security_check_node(state, _code_quality_retriever)

    def kb_shell_search_fn(state: GeneratorAgentState) -> dict:
        return kb_shell_search_node(state, _kb_shell_retriever)

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
    if has_kb_shell_search(tool_mode):
        workflow.add_node("kb_shell_search", kb_shell_search_fn)
    if has_api_lookup(tool_mode):
        workflow.add_node("api_lookup", api_lookup_fn)
    if has_api_constraint(tool_mode):
        workflow.add_node("api_constraint", api_constraint_fn)
    if has_api_alternative(tool_mode):
        workflow.add_node("api_alternative", api_alternative_fn)
    if has_tiling_calc(tool_mode):
        workflow.add_node("tiling_calc", tiling_calc_fn)
    if has_tiling_validate(tool_mode):
        workflow.add_node("tiling_validate", tiling_validate_fn)
    if has_npu_arch(tool_mode):
        workflow.add_node("npu_arch", npu_arch_fn)
    if has_code_style(tool_mode):
        workflow.add_node("code_style", code_style_fn)
    if has_security_check(tool_mode):
        workflow.add_node("security_check", security_check_fn)

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
    if has_kb_shell_search(tool_mode):
        conditional_map["kb_shell_search"] = "kb_shell_search"
    if has_api_lookup(tool_mode):
        conditional_map["api_lookup"] = "api_lookup"
    if has_api_constraint(tool_mode):
        conditional_map["api_constraint"] = "api_constraint"
    if has_api_alternative(tool_mode):
        conditional_map["api_alternative"] = "api_alternative"
    if has_tiling_calc(tool_mode):
        conditional_map["tiling_calc"] = "tiling_calc"
    if has_tiling_validate(tool_mode):
        conditional_map["tiling_validate"] = "tiling_validate"
    if has_npu_arch(tool_mode):
        conditional_map["npu_arch"] = "npu_arch"
    if has_code_style(tool_mode):
        conditional_map["code_style"] = "code_style"
    if has_security_check(tool_mode):
        conditional_map["security_check"] = "security_check"

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
    if has_kb_shell_search(tool_mode):
        workflow.add_edge("kb_shell_search", "choose_tool")
    if has_api_lookup(tool_mode):
        workflow.add_edge("api_lookup", "choose_tool")
    if has_api_constraint(tool_mode):
        workflow.add_edge("api_constraint", "choose_tool")
    if has_api_alternative(tool_mode):
        workflow.add_edge("api_alternative", "choose_tool")
    if has_tiling_calc(tool_mode):
        workflow.add_edge("tiling_calc", "choose_tool")
    if has_tiling_validate(tool_mode):
        workflow.add_edge("tiling_validate", "choose_tool")
    if has_npu_arch(tool_mode):
        workflow.add_edge("npu_arch", "choose_tool")
    if has_code_style(tool_mode):
        workflow.add_edge("code_style", "choose_tool")
    if has_security_check(tool_mode):
        workflow.add_edge("security_check", "choose_tool")

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