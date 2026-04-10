"""
Generator Agent Module: LangGraph-based intelligent kernel generation agent.

This module provides an integrated agent framework for generating Ascend C kernels
with multi-source retrieval support (KB, Web, Code RAG).

Key components:
- ToolType: Enum for individual tool types (extensible)
- AgentToolMode: FrozenSet[ToolType] for tool combinations (no combinatorial explosion)
- Predefined modes: NO_TOOL, KB_ONLY, WEB_ONLY, CODE_RAG_ONLY, KB_AND_WEB, KB_AND_CODE_RAG, WEB_AND_CODE_RAG, ALL
- Helper functions: has_tool(), has_kb(), has_web(), has_code_rag(), parse_tool_mode()
- GeneratorAgentState: State definition for the agent workflow
- build_agent_app: Build LangGraph StateGraph application
- generate_kernel_with_agent: High-level API for kernel generation

Quick usage:
    from generator.agent import (
        generate_kernel_with_agent, KernelGenerationTask,
        AgentToolMode, ToolType, NO_TOOL, KB_ONLY, ALL,
        parse_tool_mode, has_kb
    )

    # Using predefined mode
    task = KernelGenerationTask(
        language="ascendc",
        op="gelu",
        strategy_name="add_shot",
        category="activation"
    )
    result = generate_kernel_with_agent(task, ALL)

    # Using string parsing
    result = generate_kernel_with_agent(task, "kb,web")

    # Using custom combination
    custom_mode = frozenset({ToolType.KB, ToolType.CODE_RAG})
    result = generate_kernel_with_agent(task, custom_mode)
"""

from .agent_config import (
    # Core types
    AgentToolMode,
    ToolType,
    # Predefined modes (backward compatibility)
    NO_TOOL,
    KB_ONLY,
    WEB_ONLY,
    CODE_RAG_ONLY,
    KB_AND_WEB,
    KB_AND_CODE_RAG,
    WEB_AND_CODE_RAG,
    ALL,
    # Helper functions
    has_tool,
    has_kb,
    has_web,
    has_code_rag,
    parse_tool_mode,
    tool_mode_to_string,
    # LLM config
    get_llm_config_compatible,
)
from .agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS, create_initial_state
from .agent_builder import build_agent_app, create_agent
from .agent_runner import (
    generate_kernel_with_agent,
    generate_ascendc_kernel,
    KernelGenerationTask,
    AgentGenerationResult,
)

__all__ = [
    # Core types
    'AgentToolMode',
    'ToolType',
    # Predefined modes
    'NO_TOOL',
    'KB_ONLY',
    'WEB_ONLY',
    'CODE_RAG_ONLY',
    'KB_AND_WEB',
    'KB_AND_CODE_RAG',
    'WEB_AND_CODE_RAG',
    'ALL',
    # Helper functions
    'has_tool',
    'has_kb',
    'has_web',
    'has_code_rag',
    'parse_tool_mode',
    'tool_mode_to_string',
    # LLM config
    'get_llm_config_compatible',
    # State
    'GeneratorAgentState',
    'MAX_QUERY_ROUNDS',
    'create_initial_state',
    # Builder
    'build_agent_app',
    'create_agent',
    # Runner
    'generate_kernel_with_agent',
    'generate_ascendc_kernel',
    'KernelGenerationTask',
    'AgentGenerationResult',
]