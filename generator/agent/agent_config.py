"""
Agent configuration for LangGraph-based kernel generation agent.

Extends AgentToolMode to support KB, WEB, and CODE_RAG retrieval.
"""
import os
from enum import Enum


class AgentToolMode(str, Enum):
    """
    Tool mode enum for specifying which retrieval tools to use.

    Modes:
    - NO_TOOL: No retrieval, direct LLM generation (default)
    - KB_ONLY: Only KB knowledge base (API documentation)
    - WEB_ONLY: Only Web search
    - CODE_RAG_ONLY: Only Code RAG retrieval
    - KB_AND_WEB: KB + Web
    - KB_AND_CODE_RAG: KB + Code RAG
    - WEB_AND_CODE_RAG: Web + Code RAG
    - ALL: KB + Web + Code RAG (full retrieval power)
    """
    NO_TOOL = "no_tool"
    KB_ONLY = "kb_only"
    WEB_ONLY = "web_only"
    CODE_RAG_ONLY = "code_rag_only"
    KB_AND_WEB = "kb_and_web"
    KB_AND_CODE_RAG = "kb_and_code_rag"
    WEB_AND_CODE_RAG = "web_and_code_rag"
    ALL = "all"

    def has_kb(self) -> bool:
        """Check if KB knowledge base is enabled."""
        return self in (
            AgentToolMode.KB_ONLY,
            AgentToolMode.KB_AND_WEB,
            AgentToolMode.KB_AND_CODE_RAG,
            AgentToolMode.ALL,
        )

    def has_web(self) -> bool:
        """Check if Web search is enabled."""
        return self in (
            AgentToolMode.WEB_ONLY,
            AgentToolMode.KB_AND_WEB,
            AgentToolMode.WEB_AND_CODE_RAG,
            AgentToolMode.ALL,
        )

    def has_code_rag(self) -> bool:
        """Check if Code RAG is enabled."""
        return self in (
            AgentToolMode.CODE_RAG_ONLY,
            AgentToolMode.KB_AND_CODE_RAG,
            AgentToolMode.WEB_AND_CODE_RAG,
            AgentToolMode.ALL,
        )


def get_llm_config_from_env() -> dict:
    """
    Get LLM configuration from environment variables (Agent_kernel style).

    Returns dict with api_key, base_url, model.
    Raises SystemExit if XI_AI_API_KEY is not set.
    """
    api_key = os.getenv("XI_AI_API_KEY")
    if not api_key or not api_key.strip():
        # Don't raise error here, allow fallback to generator config
        return None

    base_url = os.getenv("XI_AI_BASE_URL", "https://api-2.xi-ai.cn/v1")
    model_name = os.getenv("XI_AI_MODEL", "gpt-5")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
    }


def get_llm_config_compatible() -> dict:
    """
    Get LLM configuration with fallback support.

    Priority:
    1. XI_AI_API_KEY environment variable (Agent_kernel style)
    2. generator/utils api_config.py

    Returns dict with api_key, base_url, model.
    """
    # Try Agent_kernel style first
    env_config = get_llm_config_from_env()
    if env_config:
        return env_config

    # Fallback to generator style
    try:
        from generator.utils.utils import get_client, get_default_model_from_config
        model = get_default_model_from_config() or "deepseek-chat"
        # generator uses api_config.py which is imported by get_client
        # We need to extract the config from there
        import importlib.util
        import os as _os
        api_config_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "utils", "api_config.py"
        )
        if _os.path.exists(api_config_path):
            spec = importlib.util.spec_from_file_location("api_config", api_config_path)
            api_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_config)
            return {
                "api_key": getattr(api_config, "API_KEY", ""),
                "base_url": getattr(api_config, "BASE_URL", "https://api.deepseek.com/v1"),
                "model": getattr(api_config, "MODEL", model),
            }
    except Exception:
        pass

    # Final fallback
    return {
        "api_key": "",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    }