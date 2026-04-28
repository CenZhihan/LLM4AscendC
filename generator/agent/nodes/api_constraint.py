"""API constraint check node for generator agent."""
import re
import json
import dataclasses
from typing import Dict, Any, Optional

from ..agent_state import GeneratorAgentState
from ..query_utils import extract_api_name, get_tool_args
from ..retrievers.api_doc_retriever import ApiDocRetriever


def _normalize_api_name(api_name: str) -> str:
    name = (api_name or "").strip()
    for prefix in ("AscendC::", "ascendc::"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _normalize_constraint_context(context: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in sorted((context or {}).keys()):
        value = context[key]
        if value is None or value == "":
            continue
        if isinstance(value, str):
            normalized[key] = value.strip()
        else:
            normalized[key] = value
    return normalized


def _find_cached_constraint_entry(
    state: GeneratorAgentState,
    api_name: str,
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    target_name = _normalize_api_name(api_name)
    target_context = _normalize_constraint_context(context)
    target_key = json.dumps({"api_name": target_name, "context": target_context}, sort_keys=True, ensure_ascii=False)
    for entry in reversed(state.get("tool_calls_log") or []):
        if entry.get("tool") != "api_constraint":
            continue
        cache_key = entry.get("cache_key")
        if isinstance(cache_key, str) and cache_key == target_key:
            return entry
        entry_args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        entry_name = extract_api_name(str(entry.get("query") or ""), args=entry_args)
        if _normalize_api_name(entry_name) != target_name:
            continue
        entry_context = _normalize_constraint_context(entry_args)
        entry_context.pop("api_name", None)
        if entry_context == target_context:
            return entry
    return None


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
        lines.append(f"  检查结论: {getattr(result, 'compliance_status', 'pass')}")
        lines.append(f"  符合约束: {result.is_compliant}")
        checked_context = getattr(result, "checked_context", None) or {}
        if checked_context:
            lines.append("  调用上下文:")
            for key, value in checked_context.items():
                lines.append(f"    - {key} = {value}")
        if getattr(result, "checks_performed", None):
            lines.append("  已检查项:")
            for check in result.checks_performed:
                status = str(check.get("status", "")).upper() or "INFO"
                detail = check.get("detail", "")
                lines.append(f"    - [{status}] {check.get('name', '')}: {detail}")
        if result.constraints:
            lines.append(f"  约束条件:")
            for c in result.constraints:
                lines.append(f"    - [{c.get('severity', '')}] {c.get('desc', '')}")
        if result.violations:
            for v in result.violations:
                lines.append(f"  ❌ 违反: {v}")
        else:
            lines.append("  违反项: [未发现明确违规]")
        if getattr(result, "unknowns", None):
            lines.append("  未校验项:")
            for item in result.unknowns:
                lines.append(f"    - {item}")
        if getattr(result, "source_doc", ""):
            lines.append(f"  来源: {result.source_doc}")
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
            "api_constraint_result": {"api_name": "unknown", "constraints": [], "violations": [], "suggestion": "", "is_compliant": False, "compliance_status": "insufficient_context"},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    args = get_tool_args(state)
    api_name, context = _extract_constraint_input(query, args)
    normalized_context = _normalize_constraint_context(context)
    normalized_context.pop("api_name", None)
    cached_entry = _find_cached_constraint_entry(state, api_name, normalized_context)

    round_num = state.get("query_round_count", 0) + 1
    if cached_entry is not None:
        print(f"[Round {round_num}] 工具=API约束检查(API_CONSTRAINT), API=\"{api_name}\" (cache hit)")
        cached_result = cached_entry.get("result") if isinstance(cached_entry.get("result"), dict) else {}
        return {
            "api_constraint_results": [],
            "api_constraint_result": cached_result or {"api_name": api_name},
            "query_round_count": round_num,
            "tool_calls_log": [],
        }

    result = api_retriever.check_constraints(api_name, context)

    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "api_constraint",
        "query": query or f"API: {api_name}",
        "args": args if isinstance(args, dict) else {},
        "response": display_text,
        "result": dataclasses.asdict(result),
        "cache_key": json.dumps(
            {"api_name": _normalize_api_name(api_name), "context": normalized_context},
            sort_keys=True,
            ensure_ascii=False,
        ),
    }

    print(f"[Round {round_num}] 工具=API约束检查(API_CONSTRAINT), API=\"{api_name}\"")

    return {
        "api_constraint_results": [display_text],
        "api_constraint_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
