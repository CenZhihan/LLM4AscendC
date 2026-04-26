"""NPU Architecture query node for generator agent."""
import dataclasses
from typing import Dict, Any

from ..agent_state import GeneratorAgentState
from ..query_utils import extract_chip_name, get_tool_args
from ..retrievers.npu_arch_retriever import NpuArchRetriever


def _format_for_display(result) -> str:
    """Format ChipSpecResult for display."""
    if hasattr(result, "chip_name"):
        lines = [f"NPU 架构查询: {result.chip_name}"]
        lines.append(f"  NpuArch: {result.npu_arch}")
        lines.append(f"  UB 容量: {result.ub_capacity_bytes}B")
        lines.append(f"  Vector Cores: {result.vector_core_num}")
        lines.append(f"  Cube Cores: {result.cube_core_num}")
        lines.append(f"  HBM: {result.hbm_capacity_gb}GB")
        lines.append(f"  编译宏: {result.arch_compile_macro}")
        if result.features:
            lines.append(f"  特性: {', '.join(result.features)}")
        lines.append(f"  说明: {result.details}")
        return "\n".join(lines)
    return str(result)


def npu_arch_node(
    state: GeneratorAgentState,
    npu_arch_retriever: NpuArchRetriever = None,
) -> Dict[str, Any]:
    """
    NPU architecture query node.

    Args:
        state: Current agent state
        npu_arch_retriever: Optional pre-initialized retriever

    Returns:
        Dict with npu_arch_results, npu_arch_result (structured),
        query_round_count, tool_calls_log
    """
    if npu_arch_retriever is None:
        npu_arch_retriever = NpuArchRetriever()

    query = state.get("current_query", "")
    args = get_tool_args(state)
    chip_name = extract_chip_name(
        query,
        args=args,
        known_names=npu_arch_retriever.list_chips(),
    )
    result = npu_arch_retriever.lookup_chip_spec(chip_name)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {
        "round": round_num,
        "tool": "npu_arch",
        "query": query,
        "args": args if isinstance(args, dict) else {},
        "response": display_text,
    }

    print(f"[Round {round_num}] 工具=NPU架构查询(NPU_ARCH), chip=\"{chip_name}\"")

    return {
        "npu_arch_results": [display_text],
        "npu_arch_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
