"""Shape/stride/layout validator node for generator agent."""
import dataclasses
import json
from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..query_utils import get_tool_args
from ..retrievers.shape_stride_layout_validator import ShapeStrideLayoutValidatorResult
from ..retrievers.tiling_retriever import TilingRetriever


def _extract_direction_from_query(query: str) -> str:
    text = str(query or "").lower()
    if "gm" in text and "ub" in text and "to gm" not in text and ("to ub" in text or "gm->ub" in text or "gm_to_ub" in text):
        return "GM_TO_UB"
    if "ub" in text and "gm" in text and ("to gm" in text or "ub->gm" in text or "ub_to_gm" in text):
        return "UB_TO_GM"
    if "ub" in text and ("ub->ub" in text or "ub_to_ub" in text):
        return "UB_TO_UB"
    return ""


def resolve_shape_stride_layout_validator_request(state: GeneratorAgentState) -> dict:
    query = str(state.get("current_query", "") or "")
    args = dict(get_tool_args(state))

    if "tensor_shape" not in args and "shape" in args:
        args["tensor_shape"] = args["shape"]
    if "tensor_stride" not in args and "stride" in args:
        args["tensor_stride"] = args["stride"]
    if "movement_direction" not in args and "direction" in args:
        args["movement_direction"] = args["direction"]
    if "movement_direction" not in args:
        inferred_direction = _extract_direction_from_query(query)
        if inferred_direction:
            args["movement_direction"] = inferred_direction
    if "element_dtype" not in args and "dtype" in args:
        args["element_dtype"] = args["dtype"]
    args.setdefault("query", query)
    return args


def _format_for_display(result: ShapeStrideLayoutValidatorResult) -> str:
    summary = dataclasses.asdict(result)
    ordered_keys = [
        "status",
        "reason",
        "shape_status",
        "layout_class",
        "contiguity_status",
        "is_contiguous",
        "requires_rebuild",
        "stride_status",
        "movement_status",
        "hardware_status",
        "copy_param_status",
        "confidence",
        "normalized_request",
        "inferred_layout_details",
        "violated_rules",
        "warnings",
        "suggestions",
    ]

    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, str):
            return value.replace("\n", "\\n")
        return str(value)

    lines = ["SHAPE_STRIDE_LAYOUT_VALIDATOR_SUMMARY", "summary_version=1"]
    for key in ordered_keys:
        lines.append(f"{key}={_stringify(summary.get(key))}")
    return "\n".join(lines)


def shape_stride_layout_validator_node(
    state: GeneratorAgentState,
    tiling_retriever: TilingRetriever = None,
) -> Dict[str, Any]:
    if tiling_retriever is None:
        tiling_retriever = TilingRetriever()

    query = state.get("current_query", "")
    params = resolve_shape_stride_layout_validator_request(state)
    result = tiling_retriever.plan_shape_stride_layout_validator(params)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "shape_stride_layout_validator",
        "query": query,
        "args": params,
        "response": display_text,
        "result": dataclasses.asdict(result),
    }

    print(f"[Round {round_num}] 工具=Shape/Stride/Layout校验(SHAPE_STRIDE_LAYOUT_VALIDATOR), params={params}")

    return {
        "shape_stride_layout_validator_results": [display_text],
        "shape_stride_layout_validator_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }