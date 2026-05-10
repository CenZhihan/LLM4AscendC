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


def parse_query_fallback(query: str) -> dict:
    """Parse unstructured tiling input from query as a last resort."""
    # Defaults
    params = {
        "dtype": "float",
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
    for op in ["elementwise", "reduce", "reduction", "broadcast", "normalization", "pooling", "matmul", "conversion", "random", "convolution"]:
        if op in query.lower():
            params["op_type"] = op
            break

    # Extract intermediate buffers
    match = re.search(r"(\d+)\s*(?:intermediate|中间)", query, re.IGNORECASE)
    if match:
        params["intermediate_buffers"] = int(match.group(1))

    return params


def _num_elements_from_shape(shape: Any) -> int | None:
    if not isinstance(shape, (list, tuple)) or not shape:
        return None

    total = 1
    for dim in shape:
        try:
            dim_int = int(dim)
        except (TypeError, ValueError):
            return None
        if dim_int <= 0:
            return None
        total *= dim_int
    return total


def resolve_tiling_request(state: GeneratorAgentState) -> dict:
    """Merge structured tiling inputs before falling back to query parsing."""
    query = state.get("current_query", "")
    tool_choice_args = state.get("tool_choice_json", {}).get("args")
    structured_args = dict(tool_choice_args) if isinstance(tool_choice_args, dict) else {}
    params = parse_query_fallback(query)
    params.update({
        key: value
        for key, value in structured_args.items()
        if key in {
            "total_elements",
            "dtype",
            "op_type",
            "intermediate_buffers",
            "ub_capacity_bytes",
            "op_name",
            "input_shapes",
            "input_shape",
            "output_shape",
            "permutation",
            "reduction_axes",
            "keepdim",
            "track_index",
            "output_count",
            "algorithm_hint",
            "precision_sensitive",
            "chip",
        }
        and value not in (None, "")
    })
    inferred_total_elements = _num_elements_from_shape(structured_args.get("output_shape"))
    if inferred_total_elements is None:
        inferred_total_elements = _num_elements_from_shape(structured_args.get("input_shape"))
    if inferred_total_elements is None:
        input_shapes = structured_args.get("input_shapes")
        if isinstance(input_shapes, (list, tuple)) and len(input_shapes) == 1:
            inferred_total_elements = _num_elements_from_shape(input_shapes[0])
    if inferred_total_elements is not None and structured_args.get("total_elements") in (None, ""):
        params["total_elements"] = inferred_total_elements
    params["op_name"] = str(
        structured_args.get("op_name")
        or state.get("op_name")
        or ""
    )
    params["state_category"] = str(
        structured_args.get("state_category")
        or state.get("category")
        or ""
    )
    params["query"] = query
    params.setdefault("total_elements", 0)
    return params


def _format_for_display(result) -> str:
    """Format TilingParamsResult for display with a stable summary."""
    summary = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else {"value": str(result)}
    ordered_keys = [
        "status",
        "supported",
        "operator_class",
        "strategy_kind",
        "algorithm_kind",
        "load_mode",
        "reason",
        "required_inputs",
        "block_num",
        "num_per_core",
        "tail_num_last_core",
        "tile_length",
        "repeat_times",
        "ub_usage_bytes",
        "ub_usage_pct",
        "output_count",
        "output_elements",
        "workspace_bytes",
        "group_count",
        "chunk_size",
        "chunk_count",
        "last_chunk_size",
        "tile_a0_len",
        "aligned_cols",
        "collapsed_pattern",
        "normalized_shape",
        "normalized_axes",
        "stage_summaries",
        "constraints_met",
        "formula_used",
        "warnings",
    ]

    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, list):
            return "[" + ", ".join(_stringify(item) for item in value) + "]"
        return str(value)

    lines = ["TILING_CALC_SUMMARY", "summary_version=1"]
    for key in ordered_keys:
        lines.append(f"{key}={_stringify(summary.get(key))}")
    return "\n".join(lines)


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
    params = resolve_tiling_request(state)
    result = tiling_retriever.compute_tiling(**params)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "tiling_calc", "query": query, "response": display_text}

    print(f"[Round {round_num}] 工具=Tiling计算(TILING_CALC), params={params}")

    return {
        "tiling_calc_results": [display_text],
        "tiling_calc_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
