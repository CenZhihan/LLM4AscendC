"""
Tiling calculation node for generator agent.

Computes optimal tiling parameters for operators.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.tiling_retriever import TilingRetriever


def _parse_tiling_input(query: str) -> dict:
    """Parse tiling calculation input from query."""
    # Defaults
    params = {
        "total_elements": 1024,
        "dtype": "float",
        "op_type": "elementwise",
        "intermediate_buffers": 0,
    }

    # Extract total elements
    match = re.search(r"(\d+)\s*(?:elements?|元素)", query, re.IGNORECASE)
    if match:
        params["total_elements"] = int(match.group(1))

    # Extract dtype
    for dtype in ["float", "half", "float16", "bfloat16", "bf16", "int32", "int16", "int8", "uint8"]:
        if dtype in query.lower():
            params["dtype"] = dtype
            break

    # Extract op type
    for op in ["elementwise", "reduce", "broadcast"]:
        if op in query.lower():
            params["op_type"] = op
            break

    # Extract intermediate buffers
    match = re.search(r"(\d+)\s*(?:intermediate|中间)", query, re.IGNORECASE)
    if match:
        params["intermediate_buffers"] = int(match.group(1))

    return params


def _format_for_display(result) -> str:
    """Format TilingParamsResult for display."""
    if hasattr(result, "tile_length"):
        lines = [f"Tiling 计算结果:"]
        lines.append(f"  block_num = {result.block_num}")
        lines.append(f"  num_per_core = {result.num_per_core}")
        lines.append(f"  tail_num_last_core = {result.tail_num_last_core}")
        lines.append(f"  tile_length = {result.tile_length}")
        lines.append(f"  repeat_times = {result.repeat_times}")
        lines.append(f"  UB 使用量 = {result.ub_usage_bytes}B ({result.ub_usage_pct}%)")
        lines.append(f"  约束满足 = {result.constraints_met}")
        lines.append(f"  公式: {result.formula_used}")
        if result.warnings:
            for w in result.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)
    return str(result)


def tiling_calc_node(
    state: GeneratorAgentState,
    tiling_retriever: TilingRetriever = None,
) -> Dict[str, Any]:
    """
    Tiling calculation node.

    Args:
        state: Current agent state
        tiling_retriever: Optional pre-initialized retriever

    Returns:
        Dict with tiling_calc_results, tiling_calc_result (structured),
        query_round_count, tool_calls_log
    """
    if tiling_retriever is None:
        tiling_retriever = TilingRetriever()

    query = state.get("current_query", "")
    params = _parse_tiling_input(query)
    result = tiling_retriever.compute_tiling(**params)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "TILING_CALC", "query": query, "response": display_text}

    print(f"[Round {round_num}] 工具=Tiling计算(TILING_CALC), params={params}")

    return {
        "tiling_calc_results": [display_text],
        "tiling_calc_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
