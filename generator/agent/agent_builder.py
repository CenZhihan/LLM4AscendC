"""
Agent builder: construct LangGraph StateGraph for kernel generation.

Builds a workflow with tool selection, a single dispatch node, and answer generation.
"""
from typing import Dict, Any, Optional, Union

from langgraph.graph import StateGraph, END
from openai import OpenAI

from .agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from .agent_config import (
    AgentToolMode,
    NO_TOOL,
    has_kb,
    has_web,
    has_code_rag,
    has_code_search_snippet,
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
    has_ascend_search,
    has_ascend_fetch,
    parse_tool_mode,
    get_llm_config_compatible,
)
from .builtin_tools import register_builtin_tools_for_mode, snapshot_plugin_specs
from .nodes import (
    choose_tool_node,
    tool_dispatch_node,
    answer_node,
)
from .retrievers import KBRetriever, WebRetriever, CodeRetriever, CodeSearchSnippetRetriever
from .retrievers.env_checker import EnvCheckRetriever
from .retrievers.npu_arch_retriever import NpuArchRetriever
from .retrievers.tiling_retriever import TilingRetriever
from .retrievers.api_doc_retriever import ApiDocRetriever
from .retrievers.code_quality_retriever import CodeQualityRetriever
from .retrievers.kb_shell_search import KBShellSearchRetriever
from .retrievers.ascend_docs_search_retriever import AscendDocsSearchRetriever
from .retrievers.ascend_docs_fetch_retriever import AscendDocsFetchRetriever


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


def _route_after_choose_tool(_tool_mode: AgentToolMode):
    """Route after ``choose_tool``: max rounds, parse retry loop, answer, or dispatch."""

    def router(state: GeneratorAgentState) -> str:
        rounds = state.get("query_round_count", 0)
        if rounds >= MAX_QUERY_ROUNDS:
            return "answer"
        if state.get("tool_choice_parse_failed"):
            return "choose_tool"
        action = (state.get("next_action") or "").strip()
        if action.upper() == "ANSWER":
            return "answer"
        return "tool_dispatch"

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
        tool_mode: Enabled tool keys (``frozenset`` of lowercase names).
        llm_config: Optional LLM config (api_key, base_url, model)
        kb_retriever: Optional pre-initialized KB retriever
        web_retriever: Optional pre-initialized Web retriever
        code_retriever: Optional pre-initialized Code retriever

    Returns:
        Compiled StateGraph application
    """
    config = llm_config or get_llm_config_compatible()
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    model = config["model"]

    _kb_retriever = kb_retriever or (KBRetriever() if has_kb(tool_mode) else None)
    _web_retriever = web_retriever or (WebRetriever() if has_web(tool_mode) else None)
    _code_retriever = code_retriever or (CodeRetriever() if has_code_rag(tool_mode) else None)
    _code_search_snippet_retriever = (
        CodeSearchSnippetRetriever() if has_code_search_snippet(tool_mode) else None
    )
    _env_retriever = (
        EnvCheckRetriever()
        if (
            has_env_check_env(tool_mode)
            or has_env_check_npu(tool_mode)
            or has_env_check_api(tool_mode)
        )
        else None
    )
    _npu_arch_retriever = NpuArchRetriever() if has_npu_arch(tool_mode) else None
    _tiling_retriever = (
        TilingRetriever()
        if (has_tiling_calc(tool_mode) or has_tiling_validate(tool_mode))
        else None
    )
    _api_retriever = (
        ApiDocRetriever()
        if (
            has_api_lookup(tool_mode)
            or has_api_constraint(tool_mode)
            or has_api_alternative(tool_mode)
        )
        else None
    )
    _code_quality_retriever = (
        CodeQualityRetriever()
        if (has_code_style(tool_mode) or has_security_check(tool_mode))
        else None
    )
    _kb_shell_retriever = KBShellSearchRetriever() if has_kb_shell_search(tool_mode) else None
    _ascend_search_retriever = AscendDocsSearchRetriever() if has_ascend_search(tool_mode) else None
    _ascend_fetch_retriever = AscendDocsFetchRetriever() if has_ascend_fetch(tool_mode) else None

    plugin_snapshot = snapshot_plugin_specs(tool_mode)
    register_builtin_tools_for_mode(
        tool_mode,
        client=client,
        model=model,
        kb_retriever=_kb_retriever,
        web_retriever=_web_retriever,
        code_retriever=_code_retriever,
        code_search_snippet_retriever=_code_search_snippet_retriever,
        env_retriever=_env_retriever,
        npu_arch_retriever=_npu_arch_retriever,
        tiling_retriever=_tiling_retriever,
        api_retriever=_api_retriever,
        code_quality_retriever=_code_quality_retriever,
        kb_shell_retriever=_kb_shell_retriever,
        ascend_search_retriever=_ascend_search_retriever,
        ascend_fetch_retriever=_ascend_fetch_retriever,
        plugin_snapshot=plugin_snapshot,
    )

    def choose_tool_fn(state: GeneratorAgentState) -> dict:
        return choose_tool_node(state, client, model, tool_mode)

    def tool_dispatch_fn(state: GeneratorAgentState) -> dict:
        return tool_dispatch_node(state)

    def answer_fn(state: GeneratorAgentState) -> dict:
        return answer_node(state, client, model)

    workflow = StateGraph(GeneratorAgentState)

    workflow.add_node("entry", _entry_node)
    workflow.add_node("choose_tool", choose_tool_fn)
    workflow.add_node("tool_dispatch", tool_dispatch_fn)
    workflow.add_node("answer", answer_fn)

    workflow.set_entry_point("entry")

    workflow.add_conditional_edges(
        "entry",
        _route_entry(tool_mode),
        {"answer": "answer", "choose_tool": "choose_tool"},
    )

    route_after_choose = _route_after_choose_tool(tool_mode)
    workflow.add_conditional_edges(
        "choose_tool",
        route_after_choose,
        {
            "answer": "answer",
            "choose_tool": "choose_tool",
            "tool_dispatch": "tool_dispatch",
        },
    )

    workflow.add_edge("tool_dispatch", "choose_tool")
    workflow.add_edge("answer", END)

    return workflow.compile()


def create_agent(
    tool_mode: Union[str, AgentToolMode] = "no_tool",
    **kwargs,
):
    """
    Create agent app with flexible tool mode input.

    Args:
        tool_mode: ``str`` (e.g. ``\"kb_only\"``, ``\"kb,web\"``) or ``frozenset`` of tool keys.
        **kwargs: Additional args passed to :func:`build_agent_app`

    Returns:
        Compiled StateGraph application
    """
    mode = parse_tool_mode(tool_mode)
    return build_agent_app(mode, **kwargs)
