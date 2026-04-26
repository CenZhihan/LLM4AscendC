"""API constraint check node for generator agent."""
import re
import json
import dataclasses
from typing import Dict, Any

from ..agent_state import GeneratorAgentState
from ..query_utils import extract_api_name, get_tool_args
from ..retrievers.api_doc_retriever import ApiDocRetriever


def _extract_constraint_input(query: str, args: Dict[str, Any]) -> tuple:
    """Extract API name and call context from query."""
    api_name = extract_api_name(query, args=args)
    context: Dict[str, Any] = dict(args or {})

    # Try to parse JSON-like dict for context
    match = re.search(r"\{.*\}", query, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            context.update(parsed)
        except json.JSONDecodeError:
            pass

    # Extract key=value pairs for context
    for key in ["count", "repeat_times", "ub_usage_bytes", "ub_capacity_bytes"]:
        m = re.search(rf"{key}\s*[=:]\s*(\d+)", query, re.IGNORECASE)
        if m:
            context[key] = int(m.group(1))

    # Extract dtype
    for dtype in ["float", "half", "float16", "bfloat16", "int32", "int16", "int8"]:
        if dtype in query.lower():
            context["dtype"] = dtype
            break

    # Check GM->UB flag
    if "gm" in query.lower() and ("ub" in query.lower() or "global" in query.lower()):
        context["is_gm_to_ub"] = True

    return api_name, context


def _format_for_display(result) -> str:
    """Format ApiConstraintResult for display."""
    if hasattr(result, "api_name"):
        lines = [f"API 约束检查: {result.api_name}"]
        lines.append(f"  符合约束: {result.is_compliant}")
        if result.constraints:
            lines.append(f"  约束条件:")
            for c in result.constraints:
                lines.append(f"    - [{c.get('severity', '')}] {c.get('desc', '')}")
        if result.violations:
            for v in result.violations:
                lines.append(f"  ❌ 违反: {v}")
        if result.suggestion:
            lines.append(f"  建议: {result.suggestion}")
        return "\n".join(lines)
    return str(result)


def api_constraint_node(
    state: GeneratorAgentState,
    api_retriever: ApiDocRetriever = None,
) -> Dict[str, Any]:
    """
    API constraint check node.

    Args:
        state: Current agent state
        api_retriever: Optional pre-initialized retriever

    Returns:
        Dict with api_constraint_results, api_constraint_result (structured),
        query_round_count, tool_calls_log
    """
    if api_retriever is None:
        api_retriever = ApiDocRetriever()

    if not api_retriever.is_available():
        return {
            "api_constraint_results": ["[API 文档未找到，无法检查约束]"],
            "api_constraint_result": {"api_name": "unknown", "constraints": [], "violations": [], "suggestion": "", "is_compliant": True},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    args = get_tool_args(state)
    api_name, context = _extract_constraint_input(query, args)
    result = api_retriever.check_constraints(api_name, context)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "api_constraint",
        "query": query or f"API: {api_name}",
        "args": args if isinstance(args, dict) else {},
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=API约束检查(API_CONSTRAINT), API=\"{api_name}\"")

    return {
        "api_constraint_results": [display_text],
        "api_constraint_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
