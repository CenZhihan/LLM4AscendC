"""
Security pattern check node for generator agent.

Checks code for security issues (buffer overflow, memory leaks, etc.).
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.code_quality_retriever import CodeQualityRetriever


def _extract_code(query: str) -> str:
    """Extract code from query string."""
    match = re.search(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return query


def _extract_op_type(query: str) -> str:
    """Extract operator type from query."""
    for op in ["elementwise", "reduce", "broadcast", "matmul", "convolution"]:
        if op in query.lower():
            return op
    return "elementwise"


def _format_for_display(result) -> str:
    """Format SecurityCheckResult for display."""
    if hasattr(result, "safe"):
        status = "安全" if result.safe else "发现安全隐患"
        lines = [f"安全检查: {status}"]
        if result.issues:
            for issue in result.issues[:20]:
                severity = issue.get("severity", "")
                issue_type = issue.get("type", "")
                description = issue.get("description", "")
                location = issue.get("location", "")
                lines.append(f"  [{severity}] {location} ({issue_type}): {description}")
                if issue.get("fix"):
                    lines.append(f"    修复: {issue['fix']}")
            if len(result.issues) > 20:
                lines.append(f"  ... 共 {len(result.issues)} 个问题")
        else:
            lines.append("  未发现安全问题。")
        return "\n".join(lines)
    return str(result)


def security_check_node(
    state: GeneratorAgentState,
    code_quality_retriever: CodeQualityRetriever = None,
) -> Dict[str, Any]:
    """
    Security pattern check node.

    Args:
        state: Current agent state
        code_quality_retriever: Optional pre-initialized retriever

    Returns:
        Dict with security_check_results, security_check_result (structured),
        query_round_count, tool_calls_log
    """
    if code_quality_retriever is None:
        code_quality_retriever = CodeQualityRetriever()

    query = state.get("current_query", "")
    code = _extract_code(query)
    op_type = _extract_op_type(query)
    result = code_quality_retriever.check_security(code, op_type)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "SECURITY_CHECK", "query": query[:100], "response": display_text}

    print(f"[Round {round_num}] 工具=安全检查(SECURITY_CHECK), safe={result.safe}, issues={len(result.issues)}")

    return {
        "security_check_results": [display_text],
        "security_check_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
