"""
Generator Agent Module: LangGraph-based intelligent kernel generation agent.

This module provides an integrated agent framework for generating Ascend C kernels
with multi-source retrieval support (KB, Web, Code RAG).

Key components:
- AgentToolMode: Enum for specifying which retrieval tools to use
- GeneratorAgentState: State definition for the agent workflow
- build_agent_app: Build LangGraph StateGraph application
- generate_kernel_with_agent: High-level API for kernel generation

Quick usage:
    from generator.agent import generate_kernel_with_agent, AgentToolMode, KernelGenerationTask

    task = KernelGenerationTask(
        language="ascendc",
        op="gelu",
        strategy_name="add_shot",
        category="activation"
    )
    result = generate_kernel_with_agent(task, AgentToolMode.ALL)
    print(result.generated_code)
"""

from .agent_config import AgentToolMode, get_llm_config_compatible
from .agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS, create_initial_state
from .agent_builder import build_agent_app, create_agent
from .agent_runner import (
    generate_kernel_with_agent,
    generate_ascendc_kernel,
    KernelGenerationTask,
    AgentGenerationResult,
)

__all__ = [
    # Config
    'AgentToolMode',
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