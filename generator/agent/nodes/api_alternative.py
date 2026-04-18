"""
API alternative finder node for generator agent.

Finds alternative APIs when the primary API is not suitable.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.api_doc_retriever import ApiDocRetriever


def _extract_alternative_input(query: str) -> tuple:
    """Extract API name and reason from query."""
    api_name = "unknown"
    reason = "不可用"

    # Extract API name
    m = re.search(r"(?:替代|alternative|替换)\s*(?:for\s+|的\s*)?([A-Za-z_]\w*)", query, re.IGNORECASE)
    if m:
        api_name = m.group(1)

    # Also check for "API: XXX" pattern
    if api_name == "unknown":
        m = re.search(r"API[:：]?\s*([A-Za-z_]\w*)", query, re.IGNORECASE)
        if m:
            api_name = m.group(1)

    # Extract reason
    if "性能" in query or "perf" in query.lower() or "slow" in query.lower():
        reason = "性能差"
    elif "精度" in query or "precis" in query.lower() or "accurac" in query.lower():
        reason = "精度差"
    elif "不存在" in query or "not found" in query.lower() or "not exist" in query.lower():
        reason = "不存在"

    return api_name, reason


def _format_for_display(result) -> str:
    """Format ApiAlternativeResult for display."""
    if hasattr(result, "primary_api"):
        lines = [f"API 替代方案: {result.primary_api}"]
        lines.append(f"  原因: {result.reason}")
        if result.alternatives:
            for i, alt in enumerate(result.alternatives, 1):
                lines.append(f"  方案{i}: {alt.get('api', '')}")
                lines.append(f"    步骤: {alt.get('steps', '')}")
                lines.append(f"    性能影响: {alt.get('performance_impact', '')}")
                lines.append(f"    精度影响: {alt.get('precision_impact', '')}")
        else:
            lines.append("  未找到已知替代方案")
        lines.append(f"  推荐: {result.recommended}")
        return "\n".join(lines)
    return str(result)


def api_alternative_node(
    state: GeneratorAgentState,
    api_retriever: ApiDocRetriever = None,
) -> Dict[str, Any]:
    """
    API alternative finder node.

    Args:
        state: Current agent state
        api_retriever: Optional pre-initialized retriever

    Returns:
        Dict with api_alternative_results, api_alternative_result (structured),
        query_round_count, tool_calls_log
    """
    if api_retriever is None:
        api_retriever = ApiDocRetriever()

    if not api_retriever.is_available():
        return {
            "api_alternative_results": ["[API 文档未找到，无法查找替代方案]"],
            "api_alternative_result": {"primary_api": "unknown", "alternatives": [], "recommended": "", "reason": ""},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    api_name, reason = _extract_alternative_input(query)
    result = api_retriever.find_alternatives(api_name, reason)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "api_alternative",
        "query": f"API: {api_name}",
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=API替代方案(API_ALTERNATIVE), API=\"{api_name}\", reason=\"{reason}\"")

    return {
        "api_alternative_results": [display_text],
        "api_alternative_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
