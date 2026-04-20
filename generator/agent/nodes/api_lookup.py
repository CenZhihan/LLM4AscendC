"""API signature lookup node for generator agent."""
import dataclasses
from typing import Dict, Any

from ..agent_state import GeneratorAgentState
from ..query_utils import extract_api_name, get_tool_args
from ..retrievers.api_doc_retriever import ApiDocRetriever


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
    args = get_tool_args(state)
    api_name = extract_api_name(query, args=args, known_names=api_retriever.known_api_names())
    result = api_retriever.lookup_signature(api_name)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "api_lookup",
        "query": query or f"API: {api_name}",
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=API签名查询(API_LOOKUP), API=\"{api_name}\"")

    return {
        "api_lookup_results": [display_text],
        "api_lookup_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
