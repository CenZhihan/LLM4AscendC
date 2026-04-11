"""
Agent configuration for LangGraph-based kernel generation agent.

Uses FrozenSet-based tool mode design for extensibility.
Adding new tools only requires adding a ToolType enum value, no need to define all combinations.
"""
import os
from enum import Enum
from typing import FrozenSet, Set, Union, Optional


class ToolType(str, Enum):
    """
    Individual tool type for retrieval.

    Each tool represents a retrieval source:
    - KB: Knowledge base (API documentation)
    - WEB: Web search (technical blogs, tutorials)
    - CODE_RAG: Code RAG (similar implementations from code library)

    Future tools can be easily added:
    - DOC_RAG: Documentation RAG
    - API_SEARCH: API search engine
    - STACKOVERFLOW: Stack Overflow search
    """
    KB = "kb"
    WEB = "web"
    CODE_RAG = "code_rag"
    ENV_CHECK = "env_check"


# Type alias for tool mode (FrozenSet for immutability and dict key usage)
AgentToolMode = FrozenSet[ToolType]


# ===== Predefined common modes (for backward compatibility and convenience) =====
NO_TOOL: AgentToolMode = frozenset()
KB_ONLY: AgentToolMode = frozenset({ToolType.KB})
WEB_ONLY: AgentToolMode = frozenset({ToolType.WEB})
CODE_RAG_ONLY: AgentToolMode = frozenset({ToolType.CODE_RAG})
KB_AND_WEB: AgentToolMode = frozenset({ToolType.KB, ToolType.WEB})
KB_AND_CODE_RAG: AgentToolMode = frozenset({ToolType.KB, ToolType.CODE_RAG})
WEB_AND_CODE_RAG: AgentToolMode = frozenset({ToolType.WEB, ToolType.CODE_RAG})
ALL: AgentToolMode = frozenset({ToolType.KB, ToolType.WEB, ToolType.CODE_RAG})


# ===== Helper functions =====
def has_tool(mode: AgentToolMode, tool: ToolType) -> bool:
    """Check if a specific tool is enabled in the mode."""
    return tool in mode


def has_kb(mode: AgentToolMode) -> bool:
    """Check if KB knowledge base is enabled."""
    return has_tool(mode, ToolType.KB)


def has_web(mode: AgentToolMode) -> bool:
    """Check if Web search is enabled."""
    return has_tool(mode, ToolType.WEB)


def has_code_rag(mode: AgentToolMode) -> bool:
    """Check if Code RAG is enabled."""
    return has_tool(mode, ToolType.CODE_RAG)


def has_env_check(mode: AgentToolMode) -> bool:
    """Check if Environment Check is enabled."""
    return has_tool(mode, ToolType.ENV_CHECK)


def _parse_tool_type(tool_str: str) -> Optional[ToolType]:
    """Parse a single tool string to ToolType."""
    tool_str = tool_str.lower().strip()
    # Try by value first (kb, web, code_rag)
    try:
        return ToolType(tool_str)
    except ValueError:
        pass
    # Try by name (KB, WEB, CODE_RAG)
    try:
        return ToolType[tool_str.upper()]
    except KeyError:
        pass
    return None


def parse_tool_mode(mode_str: Union[str, Set[str], FrozenSet[str], AgentToolMode]) -> AgentToolMode:
    """
    Parse tool mode from various input formats.

    Args:
        mode_str: Can be:
            - AgentToolMode (FrozenSet[ToolType]): returned directly
            - str: comma-separated tool names (e.g., "kb,web" or "all")
            - Set[str]: set of tool name strings
            - FrozenSet[str]: frozen set of tool name strings

    Returns:
        AgentToolMode (FrozenSet[ToolType])

    Examples:
        parse_tool_mode("kb") -> KB_ONLY
        parse_tool_mode("kb,web") -> KB_AND_WEB
        parse_tool_mode("all") -> ALL
        parse_tool_mode({"kb", "web"}) -> KB_AND_WEB
        parse_tool_mode(ALL) -> ALL (direct return)
    """
    # Already an AgentToolMode
    if isinstance(mode_str, frozenset):
        # Check if it's already ToolType elements
        if all(isinstance(t, ToolType) for t in mode_str):
            return mode_str
        # Convert string elements to ToolType
        return frozenset({_parse_tool_type(t) for t in mode_str if _parse_tool_type(t) is not None})

    # Set of strings
    if isinstance(mode_str, set):
        return frozenset({_parse_tool_type(t) for t in mode_str if _parse_tool_type(t) is not None})

    # String input
    if isinstance(mode_str, str):
        mode_str_lower = mode_str.lower().strip()

        # Handle predefined mode names
        predefined_modes = {
            "no_tool": NO_TOOL,
            "kb_only": KB_ONLY,
            "web_only": WEB_ONLY,
            "code_rag_only": CODE_RAG_ONLY,
            "kb_and_web": KB_AND_WEB,
            "kb_and_code_rag": KB_AND_CODE_RAG,
            "web_and_code_rag": WEB_AND_CODE_RAG,
            "all": ALL,
        }
        if mode_str_lower in predefined_modes:
            return predefined_modes[mode_str_lower]

        # Handle comma-separated tools
        tools = [t.strip() for t in mode_str.split(",") if t.strip()]
        return frozenset({_parse_tool_type(t) for t in tools if _parse_tool_type(t) is not None})

    # Fallback
    return NO_TOOL


def tool_mode_to_string(mode: AgentToolMode) -> str:
    """Convert AgentToolMode to string representation."""
    if not mode:
        return "no_tool"

    # Check predefined modes first
    predefined_map = {
        NO_TOOL: "no_tool",
        KB_ONLY: "kb_only",
        WEB_ONLY: "web_only",
        CODE_RAG_ONLY: "code_rag_only",
        KB_AND_WEB: "kb_and_web",
        KB_AND_CODE_RAG: "kb_and_code_rag",
        WEB_AND_CODE_RAG: "web_and_code_rag",
        ALL: "all",
    }
    if mode in predefined_map:
        return predefined_map[mode]

    # Custom combination
    return ",".join(sorted(t.value for t in mode))


# ===== LLM Configuration =====
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