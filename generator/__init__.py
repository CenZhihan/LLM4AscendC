"""
Generator module for LLM-based AscendC operator generation.

This module provides:
- RAG code retrieval for enhanced prompts
- Multiple prompt generation strategies
- Parallel LLM API invocation
- LangGraph-based intelligent agent for kernel generation
"""

from .dataset import dataset, category2exampleop
from .config import (
    max_tokens, temperature, top_p, num_completions, seed_num,
    rag_index_path, rag_embedding_model, rag_top_k, rag_max_chars,
    rag_code_dir, rag_file_extensions, ref_impl_base_path,
    # Agent configuration
    agent_kb_persist_dir, agent_kb_collection_name,
    agent_web_max_results, agent_web_max_fetch_urls,
    agent_code_rag_top_k, agent_max_query_rounds,
)

# Agent module exports
from .agent import (
    generate_kernel_with_agent,
    generate_ascendc_kernel,
    AgentToolMode,
    KernelGenerationTask,
    AgentGenerationResult,
    GeneratorAgentState,
    build_agent_app,
)

__all__ = [
    'dataset', 'category2exampleop',
    'max_tokens', 'temperature', 'top_p', 'num_completions', 'seed_num',
    'rag_index_path', 'rag_embedding_model', 'rag_top_k', 'rag_max_chars',
    'rag_code_dir', 'rag_file_extensions', 'ref_impl_base_path',
    # Agent configuration
    'agent_kb_persist_dir', 'agent_kb_collection_name',
    'agent_web_max_results', 'agent_web_max_fetch_urls',
    'agent_code_rag_top_k', 'agent_max_query_rounds',
    # Agent API
    'generate_kernel_with_agent', 'generate_ascendc_kernel',
    'AgentToolMode', 'KernelGenerationTask', 'AgentGenerationResult',
    'GeneratorAgentState', 'build_agent_app',
]