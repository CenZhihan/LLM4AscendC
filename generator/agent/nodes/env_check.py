"""
Environment check nodes for generator agent.

Three independent nodes for structured environment checking:
- env_check_env_node: CANN environment overview
- env_check_npu_node: NPU device query
- env_check_api_node: API compatibility check
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.env_checker import EnvCheckRetriever


def _extract_api_name(query: str) -> str:
    """Extract API name from a query string."""
    # Pattern: "check API: XXX" / "check if XXX exists" / "API XXX"
    match = re.search(r"(?:检查\s*api[:：]?\s*|check\s*(?:if\s*)?(?:api\s*)?[:：]?\s*|is\s+)(\w+)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(\w+)\s*(?:api|函数|function|算子)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    # If query looks like a single identifier, treat it as API name
    if re.match(r"^[A-Za-z_]\w{2,}$", query.strip()):
        return query.strip()
    # Fallback: use first word
    parts = query.strip().split()
    return parts[0] if parts else "unknown"


def _format_for_display(result) -> str:
    """Format any result dataclass for display as text."""
    if hasattr(result, "details"):
        # EnvCheckResult
        return f"环境检查:\n{result.details}"
    if hasattr(result, "query_type"):
        # NpuDeviceResult
        return f"NPU 查询 ({result.query_type}):\n{result.raw_output}"
    if hasattr(result, "api_name"):
        # ApiCheckResult
        lines = [f"API 检查: {result.api_name}"]
        if result.found:
            lines.append(f"  找到 {len(result.matches)} 处匹配")
            lines.append(f"  头文件: {', '.join(result.header_files)}")
            for m in result.matches[:3]:
                lines.append(f"  {m}")
        else:
            lines.append(f"  未找到: {result.summary}")
        return "\n".join(lines)
    return str(result)


def env_check_env_node(
    state: GeneratorAgentState,
    env_retriever: EnvCheckRetriever = None,
) -> Dict[str, Any]:
    """
    Environment overview node: check CANN environment configuration.

    Returns:
        Dict with env_check_results, env_check_env_result (structured),
        query_round_count, tool_calls_log
    """
    if env_retriever is None:
        env_retriever = EnvCheckRetriever()

    if not env_retriever.is_available():
        print("[WARN] CANN environment not found, env check unavailable")
        result_data = {"all_passed": False, "cann_version": "未知", "cann_home": None}
        return {
            "env_check_results": ["[CANN 环境未找到，无法执行环境检查]"],
            "env_check_env_result": result_data,
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    result = env_retriever.check_env()
    round_num = state.get("query_round_count", 0) + 1
    query = state.get("current_query", "check environment")
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "env_check_env", "query": query, "response": display_text}

    print(f"[Round {round_num}] 工具=环境检查(ENV_CHECK_ENV), 查询=\"{query[:100]}\"")

    return {
        "env_check_results": [display_text],
        "env_check_env_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }


def env_check_npu_node(
    state: GeneratorAgentState,
    env_retriever: EnvCheckRetriever = None,
) -> Dict[str, Any]:
    """
    NPU device query node: query NPU device information.

    Returns:
        Dict with env_check_results, env_check_npu_result (structured),
        query_round_count, tool_calls_log
    """
    if env_retriever is None:
        env_retriever = EnvCheckRetriever()

    if not env_retriever.is_available():
        print("[WARN] CANN environment not found, NPU query unavailable")
        result_data = {"available": False, "query_type": "info", "raw_output": "CANN 环境未找到"}
        return {
            "env_check_results": ["[CANN 环境未找到，无法查询 NPU 设备]"],
            "env_check_npu_result": result_data,
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    result = env_retriever.query_npu_devices()
    round_num = state.get("query_round_count", 0) + 1
    query = state.get("current_query", "query npu devices")
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "env_check_npu", "query": query, "response": display_text}

    print(f"[Round {round_num}] 工具=NPU查询(ENV_CHECK_NPU), 查询=\"{query[:100]}\"")

    return {
        "env_check_results": [display_text],
        "env_check_npu_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }


def env_check_api_node(
    state: GeneratorAgentState,
    env_retriever: EnvCheckRetriever = None,
) -> Dict[str, Any]:
    """
    API compatibility check node: verify if an API exists in CANN headers.

    Returns:
        Dict with env_check_results, env_check_api_result (structured),
        query_round_count, tool_calls_log
    """
    if env_retriever is None:
        env_retriever = EnvCheckRetriever()

    if not env_retriever.is_available():
        print("[WARN] CANN environment not found, API check unavailable")
        result_data = {"found": False, "api_name": "unknown", "header_files": [], "matches": [], "summary": "CANN 环境未找到"}
        return {
            "env_check_results": ["[CANN 环境未找到，无法执行 API 检查]"],
            "env_check_api_result": result_data,
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    api_name = _extract_api_name(query)
    result = env_retriever.check_api_exists(api_name)
    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "env_check_api",
        "query": f"API: {api_name}",
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=API检查(ENV_CHECK_API), API=\"{api_name}\"")

    return {
        "env_check_results": [display_text],
        "env_check_api_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
