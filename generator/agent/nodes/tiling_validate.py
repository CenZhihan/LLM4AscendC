"""
Tiling validation node for generator agent.

Validates tiling parameters against hardware constraints.
"""
import re
import json
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.tiling_retriever import TilingRetriever


def _extract_tiling_params(query: str) -> dict:
    """Extract tiling parameters from query string."""
    params = {
        "tile_length": 0,
        "repeat_times": 0,
        "ub_usage_bytes": 0,
        "block_num": 1,
        "dtype": "float",
    }

    # Try to parse JSON-like dict
    match = re.search(r"\{.*\}", query, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            params.update(parsed)
            return params
        except json.JSONDecodeError:
            pass

    # Extract key=value pairs
    for key in ["tile_length", "repeat_times", "ub_usage_bytes", "block_num"]:
        m = re.search(rf"{key}\s*[=:]\s*(\d+)", query, re.IGNORECASE)
        if m:
            params[key] = int(m.group(1))

    # Extract dtype
    for dtype in ["float", "half", "float16", "bfloat16", "int32", "int16", "int8"]:
        if dtype in query.lower():
            params["dtype"] = dtype
            break

    return params


def _format_for_display(result) -> str:
    """Format TilingValidationResult for display."""
    if hasattr(result, "is_valid"):
        lines = [f"Tiling 验证结果: {'通过' if result.is_valid else '失败'}"]
        for check_name, passed in result.checks.items():
            status = "通过" if passed else "失败"
            lines.append(f"  [{status}] {check_name}")
        if result.errors:
            for e in result.errors:
                lines.append(f"  ❌ 错误: {e}")
        if result.warnings:
            for w in result.warnings:
                lines.append(f"  ⚠ 警告: {w}")
        return "\n".join(lines)
    return str(result)


def tiling_validate_node(
    state: GeneratorAgentState,
    tiling_retriever: TilingRetriever = None,
) -> Dict[str, Any]:
    """
    Tiling validation node.

    Args:
        state: Current agent state
        tiling_retriever: Optional pre-initialized retriever

    Returns:
        Dict with tiling_validate_results, tiling_validate_result (structured),
        query_round_count, tool_calls_log
    """
    if tiling_retriever is None:
        tiling_retriever = TilingRetriever()

    query = state.get("current_query", "")
    params = _extract_tiling_params(query)
    result = tiling_retriever.validate_tiling(params)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "tiling_validate",
        "query": query,
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=Tiling验证(TILING_VALIDATE), params={params}")

    return {
        "tiling_validate_results": [display_text],
        "tiling_validate_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
