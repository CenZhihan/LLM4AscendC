"""
Agent configuration for LangGraph-based kernel generation agent.

Tool modes are frozensets of canonical tool keys (lowercase snake_case).
All tools (retrieval, checks, plugins) register the same way via tool_registry.
"""
import os
from enum import Enum
from typing import FrozenSet, Iterable, List, Set, Union, Optional


# Every built-in tool key the graph may register (subset selected by AgentToolMode).
BUILTIN_TOOL_NAMES: FrozenSet[str] = frozenset(
    {
        "kb",
        "web",
        "code_rag",
        "env_check_env",
        "env_check_npu",
        "env_check_api",
        "kb_shell_search",
        "api_lookup",
        "api_constraint",
        "api_alternative",
        "tiling_calc",
        "tiling_validate",
        "npu_arch",
        "code_style",
        "security_check",
        "ascend_search",
        "ascend_fetch",
    }
)

AgentToolMode = FrozenSet[str]


class AgentToolModeEnum(str, Enum):
    """Enum for tools/generate_ascendc_operators.py argparse only."""

    NO_TOOL = "no_tool"
    KB_ONLY = "kb_only"
    WEB_ONLY = "web_only"
    KB_AND_WEB = "kb_and_web"


NO_TOOL: AgentToolMode = frozenset()
KB_ONLY: AgentToolMode = frozenset({"kb"})
WEB_ONLY: AgentToolMode = frozenset({"web"})
CODE_RAG_ONLY: AgentToolMode = frozenset({"code_rag"})
KB_AND_WEB: AgentToolMode = frozenset({"kb", "web"})
KB_AND_CODE_RAG: AgentToolMode = frozenset({"kb", "code_rag"})
WEB_AND_CODE_RAG: AgentToolMode = frozenset({"web", "code_rag"})
ALL: AgentToolMode = frozenset(
    {
        "kb",
        "web",
        "code_rag",
        "env_check_env",
        "env_check_npu",
        "env_check_api",
    }
)

AGENT_TOOL_MODE_MAP = {
    AgentToolModeEnum.NO_TOOL: NO_TOOL,
    AgentToolModeEnum.KB_ONLY: KB_ONLY,
    AgentToolModeEnum.WEB_ONLY: WEB_ONLY,
    AgentToolModeEnum.KB_AND_WEB: KB_AND_WEB,
}


def has_tool(mode: AgentToolMode, tool: str) -> bool:
    return tool.strip().lower() in mode


def has_kb(mode: AgentToolMode) -> bool:
    return "kb" in mode


def has_web(mode: AgentToolMode) -> bool:
    return "web" in mode


def has_code_rag(mode: AgentToolMode) -> bool:
    return "code_rag" in mode


def has_env_check_env(mode: AgentToolMode) -> bool:
    return "env_check_env" in mode


def has_env_check_npu(mode: AgentToolMode) -> bool:
    return "env_check_npu" in mode


def has_env_check_api(mode: AgentToolMode) -> bool:
    return "env_check_api" in mode


def has_kb_shell_search(mode: AgentToolMode) -> bool:
    return "kb_shell_search" in mode


def has_api_lookup(mode: AgentToolMode) -> bool:
    return "api_lookup" in mode


def has_api_constraint(mode: AgentToolMode) -> bool:
    return "api_constraint" in mode


def has_api_alternative(mode: AgentToolMode) -> bool:
    return "api_alternative" in mode


def has_tiling_calc(mode: AgentToolMode) -> bool:
    return "tiling_calc" in mode


def has_tiling_validate(mode: AgentToolMode) -> bool:
    return "tiling_validate" in mode


def has_npu_arch(mode: AgentToolMode) -> bool:
    return "npu_arch" in mode


def has_code_style(mode: AgentToolMode) -> bool:
    return "code_style" in mode


def has_security_check(mode: AgentToolMode) -> bool:
    return "security_check" in mode


def has_ascend_search(mode: AgentToolMode) -> bool:
    return "ascend_search" in mode


def has_ascend_fetch(mode: AgentToolMode) -> bool:
    return "ascend_fetch" in mode


def iter_tools(mode: AgentToolMode) -> Iterable[str]:
    """Yield enabled tool keys (same as iterating the mode)."""
    return iter(sorted(mode))


def iter_plugin_tools(mode: AgentToolMode) -> List[str]:
    """Return plugin tool keys present in ``mode`` (not built-in names)."""
    return sorted(t for t in mode if t not in BUILTIN_TOOL_NAMES)


def has_plugin(mode: AgentToolMode, raw_key: str) -> bool:
    """True if ``raw_key`` refers to a non-builtin tool key that is enabled in ``mode``."""
    x = normalize_tool_choice_name(raw_key or "")
    if x is None or x == "answer":
        return False
    return x in mode and x not in BUILTIN_TOOL_NAMES


def _normalize_tool_token(tok: str) -> str:
    return tok.strip().lower()


def _resolve_mode_token(tok: str) -> str:
    t = _normalize_tool_token(tok)
    if not t:
        raise ValueError("empty tool token")
    if t in BUILTIN_TOOL_NAMES:
        return t
    from .tool_registry import get_tool_registry

    if get_tool_registry().is_registered(t):
        return t
    raise ValueError(
        f"unknown tool {tok!r}; use a built-in key (e.g. kb, web) or register a plugin before parse_tool_mode"
    )


def parse_tool_mode(mode_str: Union[str, Set[str], FrozenSet[str], AgentToolMode]) -> AgentToolMode:
    if isinstance(mode_str, frozenset):
        if not mode_str:
            return NO_TOOL
        out: List[str] = []
        for t in mode_str:
            if isinstance(t, str):
                out.append(_resolve_mode_token(t))
            else:
                raise TypeError(f"Invalid tool mode element type: {type(t)}")
        return frozenset(out)

    if isinstance(mode_str, set):
        if not mode_str:
            return NO_TOOL
        return frozenset(_resolve_mode_token(str(x)) for x in mode_str)

    if isinstance(mode_str, str):
        mode_str_lower = mode_str.lower().strip()
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
        tools = [t.strip() for t in mode_str.split(",") if t.strip()]
        if not tools:
            return NO_TOOL
        return frozenset(_resolve_mode_token(t) for t in tools)

    return NO_TOOL


def tool_mode_to_string(mode: AgentToolMode) -> str:
    if not mode:
        return "no_tool"
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
    return ",".join(sorted(mode))


def normalize_tool_choice_name(raw: str) -> Optional[str]:
    """
    Map model output for ``tool`` field to canonical key or 'answer'.

    Accepts kb, KB, legacy ENV_CHECK_ENV, etc.
    """
    s = (raw or "").strip()
    if not s:
        return None
    if s.upper() == "ANSWER":
        return "answer"
    key = s.lower().replace("-", "_")
    # Legacy uppercase enum-style names -> lowercase keys
    legacy = {
        "kb": "kb",
        "web": "web",
        "code_rag": "code_rag",
        "env_check_env": "env_check_env",
        "env_check_npu": "env_check_npu",
        "env_check_api": "env_check_api",
        "kb_shell_search": "kb_shell_search",
        "api_lookup": "api_lookup",
        "api_constraint": "api_constraint",
        "api_alternative": "api_alternative",
        "tiling_calc": "tiling_calc",
        "tiling_validate": "tiling_validate",
        "npu_arch": "npu_arch",
        "code_style": "code_style",
        "security_check": "security_check",
        "ascend_search": "ascend_search",
        "ascend_fetch": "ascend_fetch",
    }
    u = s.upper()
    legacy_upper = {
        "KB": "kb",
        "WEB": "web",
        "CODE_RAG": "code_rag",
        "ENV_CHECK_ENV": "env_check_env",
        "ENV_CHECK_NPU": "env_check_npu",
        "ENV_CHECK_API": "env_check_api",
        "KB_SHELL_SEARCH": "kb_shell_search",
        "API_LOOKUP": "api_lookup",
        "API_CONSTRAINT": "api_constraint",
        "API_ALTERNATIVE": "api_alternative",
        "TILING_CALC": "tiling_calc",
        "TILING_VALIDATE": "tiling_validate",
        "NPU_ARCH": "npu_arch",
        "CODE_STYLE": "code_style",
        "SECURITY_CHECK": "security_check",
        "ASCEND_SEARCH": "ascend_search",
        "ASCEND_FETCH": "ascend_fetch",
    }
    if u in legacy_upper:
        return legacy_upper[u]
    if key in legacy:
        return legacy[key]
    return key


# ===== LLM Configuration =====
def _load_llm_config_from_local_api_file() -> dict:
    """仅从 ``generator/local_api_config.py`` 读取 api_key / base_url / model（不含环境变量）。"""
    import importlib.util

    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_config_path = os.path.join(_project_root, "generator", "local_api_config.py")
    if not os.path.exists(local_config_path):
        raise FileNotFoundError(
            "Agent 需要 LLM 配置：请创建 generator/local_api_config.py（可复制 "
            "generator/local_api_config.example.py）。不再从环境变量读取。"
        )
    spec = importlib.util.spec_from_file_location("local_api_config", local_config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 LLM 配置文件: {local_config_path}")
    lac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lac)
    key = (getattr(lac, "XI_AI_API_KEY", None) or getattr(lac, "OPENAI_API_KEY", None) or "").strip()
    base = (getattr(lac, "XI_AI_BASE_URL", None) or getattr(lac, "OPENAI_API_BASE", None) or "").strip()
    model = (getattr(lac, "XI_AI_MODEL", None) or getattr(lac, "MODEL", None) or "").strip()
    if not key:
        raise ValueError(
            "generator/local_api_config.py 中未设置有效的 XI_AI_API_KEY 或 OPENAI_API_KEY。"
        )
    return {
        "api_key": key,
        "base_url": base or "https://api-2.xi-ai.cn/v1",
        "model": model,
    }


def get_llm_config_compatible(cli_model: Optional[str] = None) -> dict:
    """
    解析 Agent 使用的 LLM 配置。

    **model** 优先级（仅这两处，不再读环境变量）：
    1. ``cli_model`` 非空时作为 ``model``；
    2. 否则使用 ``local_api_config.py`` 中的 ``XI_AI_MODEL`` / ``MODEL``。

    ``api_key``、``base_url`` 仅从 ``generator/local_api_config.py`` 读取。
    """
    cfg = _load_llm_config_from_local_api_file()
    if cli_model is not None and str(cli_model).strip():
        cfg = {**cfg, "model": str(cli_model).strip()}
    if not (cfg.get("model") or "").strip():
        raise ValueError(
            "未设置 model：请在命令行传入 --model，或在 generator/local_api_config.py 中设置 "
            "XI_AI_MODEL / MODEL。"
        )
    return cfg


def model_slug_for_path(model: str) -> str:
    """将 model 名转为安全的单段目录名（用于 ``output/ascendc/<slug>/...``）。"""
    s = (model or "").strip() or "unknown"
    for ch in r'/\:*?"<>| ':
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    s = s.strip("-")
    return s or "unknown"
