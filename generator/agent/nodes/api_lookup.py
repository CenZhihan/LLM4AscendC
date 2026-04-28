"""API signature lookup node for generator agent."""
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


def _find_cached_lookup_entry(state: GeneratorAgentState, api_name: str) -> Optional[Dict[str, Any]]:
    target = _normalize_api_name(api_name)
    for entry in reversed(state.get("tool_calls_log") or []):
        if entry.get("tool") != "api_lookup":
            continue
        cache_key = entry.get("cache_key")
        if isinstance(cache_key, str) and cache_key == target:
            return entry
        entry_args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        entry_name = extract_api_name(str(entry.get("query") or ""), args=entry_args)
        if _normalize_api_name(entry_name) == target:
            return entry
    return None


def _format_for_display(result) -> str:
    """Format ApiSignatureResult for display."""
    if hasattr(result, "api_name"):
        lines = [f"API 签名查询: {result.api_name}"]
        lines.append(f"  结果类型: {getattr(result, 'match_kind', 'unknown')}")
        lines.append(f"  置信度: {getattr(result, 'confidence', 'low')}")
        lines.append(f"  可直接用于生成: {bool(getattr(result, 'is_actionable', False))}")
        if result.signature:
            lines.append(f"  签名: {result.signature}")
        else:
            lines.append("  签名: [未直接解析到签名，见下方头文件和文档线索]")
        if result.params:
            lines.append("  参数:")
            for param in result.params[:8]:
                direction = param.get("dir") or param.get("direction") or ""
                suffix = f" ({direction})" if direction else ""
                lines.append(f"    - {param.get('name', '?')}: {param.get('type', '?')}{suffix}")
        header_files = list(getattr(result, "header_files", None) or [])
        if header_files:
            preview = ", ".join(header_files[:5])
            more = "" if len(header_files) <= 5 else f" 等 {len(header_files)} 个"
            lines.append(f"  头文件: {preview}{more}")
        if result.supported_dtypes:
            lines.append(f"  支持类型: {', '.join(result.supported_dtypes)}")
        if result.repeat_times_limit is not None:
            lines.append(f"  repeatTimes 限制: {result.repeat_times_limit}")
        if result.example_call:
            lines.append(f"  调用示例: {result.example_call}")
        lines.append(f"  来源: {result.source_doc}")
        if getattr(result, "doc_metadata", None):
            lines.append("  相关文档:")
            for item in result.doc_metadata[:3]:
                section = item.get("section") or item.get("title") or ""
                excerpt = " ".join(str(item.get("excerpt") or "").split())
                line = f"    - {item.get('path', '')}"
                if section:
                    line += f" | {section}"
                if excerpt:
                    line += f" | {excerpt}"
                lines.append(line)
        if result.details:
            lines.append("  检索摘要:")
            for detail_line in str(result.details).splitlines()[:8]:
                lines.append(f"    {detail_line}")
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
            "api_lookup_result": {"api_name": "unknown", "signature": "", "supported_dtypes": [], "repeat_times_limit": None, "match_kind": "not_found", "confidence": "low", "is_actionable": False},
            "query_round_count": state.get("query_round_count", 0) + 1,
            "tool_calls_log": [],
        }

    query = state.get("current_query", "")
    args = get_tool_args(state)
    api_name = extract_api_name(query, args=args, known_names=api_retriever.known_api_names())
    cached_entry = _find_cached_lookup_entry(state, api_name)

    round_num = state.get("query_round_count", 0) + 1
    if cached_entry is not None:
        print(f"[Round {round_num}] 工具=API签名查询(API_LOOKUP), API=\"{api_name}\" (cache hit)")
        cached_result = cached_entry.get("result") if isinstance(cached_entry.get("result"), dict) else {}
        return {
            "api_lookup_results": [],
            "api_lookup_result": cached_result or {"api_name": api_name},
            "query_round_count": round_num,
            "tool_calls_log": [],
        }

    result = api_retriever.lookup_signature(api_name)

    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "api_lookup",
        "query": query or f"API: {api_name}",
        "args": args if isinstance(args, dict) else {},
        "response": display_text,
        "result": dataclasses.asdict(result),
        "cache_key": _normalize_api_name(api_name),
    }

    print(f"[Round {round_num}] 工具=API签名查询(API_LOOKUP), API=\"{api_name}\"")

    return {
        "api_lookup_results": [display_text],
        "api_lookup_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
