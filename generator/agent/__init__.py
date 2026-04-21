"""
Generator Agent Module: LangGraph-based intelligent kernel generation agent.

This module provides an integrated agent framework for generating Ascend C kernels
with multi-source retrieval support (KB, Web, Code RAG, and many auxiliary tools).

Key components:
- ``AgentToolMode``: ``frozenset`` of canonical lowercase tool keys (and optional plugin keys)
- Predefined modes: ``NO_TOOL``, ``KB_ONLY``, ``WEB_ONLY``, ``CODE_RAG_ONLY``, ``KB_AND_WEB``, …
- Helpers: ``has_tool()``, ``has_kb()``, ``has_plugin()``, ``iter_plugin_tools()``, ``parse_tool_mode()``,
  ``normalize_tool_choice_name()``
- Tool selection JSON: see ``generator.agent.tool_choice.ToolChoiceV1``
- Registry: ``register_tool()`` / ``get_tool_registry()`` — built-ins are registered inside ``build_agent_app``
- ``GeneratorAgentState``: workflow state
- ``build_agent_app`` / ``create_agent``: compile the LangGraph app
- ``generate_kernel_with_agent``: high-level API

Quick usage::

    from generator.agent import (
        generate_kernel_with_agent, KernelGenerationTask,
        AgentToolMode, NO_TOOL, ALL, parse_tool_mode, has_kb,
    )

    task = KernelGenerationTask(
        language="ascendc", op="gelu", strategy_name="add_shot", category="activation"
    )
    result = generate_kernel_with_agent(task, ALL)
    result = generate_kernel_with_agent(task, "kb,web")
    custom_mode = frozenset({"kb", "code_rag"})
    result = generate_kernel_with_agent(task, custom_mode)

    from generator.agent.tool_registry import register_tool, RegisteredToolSpec

    register_tool(
        RegisteredToolSpec(
            name="my_tool",
            display_name="My",
            description="...",
            parameter_docs="...",
            handler=lambda state: {"registered_tool_results": ["ok"]},
            examples=[],
        )
    )
    result = generate_kernel_with_agent(task, "kb,my_tool")
"""

from .agent_config import (
    AgentToolMode,
    NO_TOOL,
    KB_ONLY,
    WEB_ONLY,
    CODE_RAG_ONLY,
    KB_AND_WEB,
    KB_AND_CODE_RAG,
    WEB_AND_CODE_RAG,
    ALL,
    has_tool,
    has_kb,
    has_web,
    has_code_rag,
    has_code_search_snippet,
    parse_tool_mode,
    tool_mode_to_string,
    get_llm_config_compatible,
    has_plugin,
    iter_plugin_tools,
    normalize_tool_choice_name,
)
from .tool_registry import register_tool, get_tool_registry, RegisteredToolSpec
from .tool_choice import ToolChoiceV1


def __getattr__(name: str):
    """Lazy-import LangGraph-heavy submodules so lightweight imports (e.g. tests) work without langgraph."""
    if name in ("GeneratorAgentState", "MAX_QUERY_ROUNDS", "create_initial_state"):
        from . import agent_state

        return getattr(agent_state, name)
    if name in ("build_agent_app", "create_agent"):
        from . import agent_builder

        return getattr(agent_builder, name)
    if name in (
        "generate_kernel_with_agent",
        "generate_ascendc_kernel",
        "KernelGenerationTask",
        "AgentGenerationResult",
    ):
        from . import agent_runner

        return getattr(agent_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentToolMode",
    "NO_TOOL",
    "KB_ONLY",
    "WEB_ONLY",
    "CODE_RAG_ONLY",
    "KB_AND_WEB",
    "KB_AND_CODE_RAG",
    "WEB_AND_CODE_RAG",
    "ALL",
    "has_tool",
    "has_kb",
    "has_web",
    "has_code_rag",
    "has_code_search_snippet",
    "parse_tool_mode",
    "tool_mode_to_string",
    "has_plugin",
    "iter_plugin_tools",
    "normalize_tool_choice_name",
    "register_tool",
    "get_tool_registry",
    "RegisteredToolSpec",
    "ToolChoiceV1",
    "get_llm_config_compatible",
    "GeneratorAgentState",
    "MAX_QUERY_ROUNDS",
    "create_initial_state",
    "build_agent_app",
    "create_agent",
    "generate_kernel_with_agent",
    "generate_ascendc_kernel",
    "KernelGenerationTask",
    "AgentGenerationResult",
]
