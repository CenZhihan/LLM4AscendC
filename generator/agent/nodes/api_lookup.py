"""
API signature lookup node for generator agent.

Queries API signatures, supported dtypes, and repeatTimes limits.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.api_doc_retriever import ApiDocRetriever


def _extract_api_name(query: str) -> str:
    """Extract API name from query string."""
    # Pattern: "lookup XXX" / "API: XXX" / "signature of XXX"
    match = re.search(r"(?:lookup|API[:：]?\s*|signature\s+(?:of|for)\s+)([A-Za-z_]\w*)", query, re.IGNORECASE)
    if match:
        return match.group(1)

    # Check for known API patterns (e.g., AscendC::DataCopy)
    match = re.search(r"(AscendC::[A-Za-z_]\w*)", query)
    if match:
        return match.group(1)

    # Single identifier
    match = re.search(r"\b([A-Z][A-Za-z_]\w{2,})\b", query)
    if match:
        return match.group(1)

    return "unknown"


def _format_for_display(result) -> str:
    """Format ApiSignatureResult for display."""
    if hasattr(result, "api_name"):
        lines = [f"API 签名查询: {result.api_name}"]
        if result.signature:
            lines.append(f"  签名: {result.signature}")
        if result.supported_dtypes:
            lines.append(f"  支持类型: {', '.join(result.supported_dtypes)}")
        if result.repeat_times_limit is not None:
            lines.append(f"  repeatTimes 限制: {result.repeat_times_limit}")
        if result.example_call:
            lines.append(f"  调用示例: {result.example_call}")
        lines.append(f"  来源: {result.source_doc}")
        if result.details:
            lines.append(f"  详情: {result.details}")
        return "\n".join(lines)
    return str(result)


def api_lookup_node(
    state: GeneratorAgentState,
    api_retriever: ApiDocRetriever = None,
) -> Dict[str, Any]:
    """
    API signature lookup node.

    Args:
        state: Current agent state
        api_retriever: Optional pre-initialized retriever

    Returns:
        Dict with api_lookup_results, api_lookup_result (structured),
        query_round_count, tool_calls_log
    """
    if api_retriever is None:
        api_retriever = ApiDocRetriever()

    if not api_retriever.is_available():
        print("[WARN] API docs not found, api lookup unavailable")
        return {
            "api_lookup_results": ["[API 文档未找到，无法查询签名]"],
            "api_lookup_result": {"api_name": "unknown", "signature": "", "supported_dtypes": [], "repeat_times_limit": None},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    api_name = _extract_api_name(query)
    result = api_retriever.lookup_signature(api_name)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "API_LOOKUP", "query": f"API: {api_name}", "response": display_text}

    print(f"[Round {round_num}] 工具=API签名查询(API_LOOKUP), API=\"{api_name}\"")

    return {
        "api_lookup_results": [display_text],
        "api_lookup_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
