"""
Code style check node for generator agent.

Checks code against Ascend C coding conventions.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.code_quality_retriever import CodeQualityRetriever


def _extract_code(query: str) -> str:
    """Extract code from query string."""
    # Try to find code block between ``` markers
    match = re.search(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", query, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Return the whole query if no code block found
    return query


def _format_for_display(result) -> str:
    """Format CodingStyleResult for display."""
    if hasattr(result, "passed"):
        status = "通过" if result.passed else "未通过"
        lines = [f"代码风格检查: {status} (得分: {result.score}/100)"]
        if result.issues:
            for issue in result.issues[:20]:  # Limit display
                severity = issue.get("severity", "")
                rule = issue.get("rule", "")
                message = issue.get("message", "")
                line_num = issue.get("line", "?")
                lines.append(f"  [{severity}] 行{line_num} ({rule}): {message}")
                if issue.get("suggestion"):
                    lines.append(f"    建议: {issue['suggestion']}")
            if len(result.issues) > 20:
                lines.append(f"  ... 共 {len(result.issues)} 个问题")
        else:
            lines.append("  未发现风格问题。")
        return "\n".join(lines)
    return str(result)


def code_style_node(
    state: GeneratorAgentState,
    code_quality_retriever: CodeQualityRetriever = None,
) -> Dict[str, Any]:
    """
    Code style check node.

    Args:
        state: Current agent state
        code_quality_retriever: Optional pre-initialized retriever

    Returns:
        Dict with code_style_results, code_style_result (structured),
        query_round_count, tool_calls_log
    """
    if code_quality_retriever is None:
        code_quality_retriever = CodeQualityRetriever()

    query = state.get("current_query", "")
    code = _extract_code(query)
    result = code_quality_retriever.check_style(code)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "CODE_STYLE", "query": query[:100], "response": display_text}

    print(f"[Round {round_num}] 工具=代码风格检查(CODE_STYLE), issues={len(result.issues)}, score={result.score}")

    return {
        "code_style_results": [display_text],
        "code_style_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
