"""
Tiling budget/codegen node for generator agent.

Builds an alignment-aware UB budget plan and queue init code.
"""
import dataclasses
import json
from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..retrievers.tiling_budget_codegen import TilingBudgetCodegenResult
from ..retrievers.tiling_retriever import TilingRetriever


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


def resolve_tiling_budget_codegen_request(state: GeneratorAgentState) -> dict:
    query = str(state.get("current_query", "") or "")
    tool_choice_args = state.get("tool_choice_json", {}).get("args")
    structured_args = dict(tool_choice_args) if isinstance(tool_choice_args, dict) else {}

    inferred_total_elements = _num_elements_from_shape(structured_args.get("total_shape"))
    if inferred_total_elements is None:
        inferred_total_elements = _num_elements_from_shape(structured_args.get("output_shape"))
    if inferred_total_elements is None:
        inferred_total_elements = _num_elements_from_shape(structured_args.get("input_shape"))

    if inferred_total_elements is not None and structured_args.get("total_elements") in (None, ""):
        structured_args["total_elements"] = inferred_total_elements

    structured_args.setdefault("op_name", str(state.get("op_name") or ""))
    structured_args.setdefault("state_category", str(state.get("category") or ""))
    structured_args.setdefault("query", query)
    return structured_args


def _format_for_display(result: TilingBudgetCodegenResult) -> str:
    summary = dataclasses.asdict(result)
    ordered_keys = [
        "status",
        "reason",
        "required_inputs",
        "supported",
        "operator_class",
        "strategy_kind",
        "algorithm_kind",
        "load_mode",
        "block_num",
        "seed_status",
        "seed_strategy_kind",
        "block_dim",
        "tile_length",
        "loop_count",
        "tail_length",
        "num_per_core",
        "last_core_num",
        "last_core_loop_count",
        "last_core_tail_length",
        "tail_num_last_core",
        "repeat_times",
        "stage_total_bytes",
        "ub_reserved_bytes",
        "ub_total_bytes",
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
        "formula_used",
        "constraints_met",
        "warnings",
        "planning_validation_status",
        "planning_validation_errors",
        "planning_validation_warnings",
        "hardware_validation_status",
        "hardware_validation_errors",
        "hardware_validation_warnings",
        "strategy_suggestions",
        "ub_budget_table",
        "init_code",
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

    lines = ["TILING_BUDGET_CODEGEN_SUMMARY", "summary_version=1"]
    for key in ordered_keys:
        lines.append(f"{key}={_stringify(summary.get(key))}")
    return "\n".join(lines)


def tiling_budget_codegen_node(
    state: GeneratorAgentState,
    tiling_retriever: TilingRetriever = None,
) -> Dict[str, Any]:
    if tiling_retriever is None:
        tiling_retriever = TilingRetriever()

    query = state.get("current_query", "")
    params = resolve_tiling_budget_codegen_request(state)
    result = tiling_retriever.plan_tiling_budget_codegen(params)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "tiling_budget_codegen",
        "query": query,
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=Tiling预算/代码生成(TILING_BUDGET_CODEGEN), params={params}")

    return {
        "tiling_budget_codegen_results": [display_text],
        "tiling_budget_codegen_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }