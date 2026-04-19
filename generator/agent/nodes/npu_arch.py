"""
NPU Architecture query node for generator agent.

Queries chip specifications from the built-in database.
"""
import re
import dataclasses
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.npu_arch_retriever import NpuArchRetriever


def _extract_chip_name(query: str) -> str:
    """Extract chip name from query string."""
    # Pattern: "query XXX chip" / "chip: XXX" / "Ascend910B2"
    match = re.search(r"(?:chip[:：]?\s*|芯片[:：]?\s*)(\w+)", query, re.IGNORECASE)
    if match:
        return match.group(1)

    # Check for known chip names
    chip_patterns = [
        r"(Ascend910B2?)", r"(Ascend910_93)", r"(Ascend910C)",
        r"(Ascend910)\b", r"(Ascend310[PB])", r"(Ascend950[DP]T)",
        r"(910B2?)", r"(910_93)", r"(910C)", r"(910)\b",
        r"(310[PB])", r"(950[DP]T)",
    ]
    for pat in chip_patterns:
        match = re.search(pat, query, re.IGNORECASE)
        if match:
            return match.group(1)

    # Default
    return "Ascend910B2"


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
    chip_name = _extract_chip_name(query)
    result = npu_arch_retriever.lookup_chip_spec(chip_name)

    round_num = state.get("query_round_count", 0) + 1
    display_text = _format_for_display(result)
    log_entry = {"round": round_num, "tool": "npu_arch", "query": query, "response": display_text}

    print(f"[Round {round_num}] 工具=NPU架构查询(NPU_ARCH), chip=\"{chip_name}\"")

    return {
        "npu_arch_results": [display_text],
        "npu_arch_result": dataclasses.asdict(result),
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
