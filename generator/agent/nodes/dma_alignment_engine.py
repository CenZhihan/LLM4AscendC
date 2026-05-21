"""dma_alignment_engine built-in: DMA alignment & DataCopy vs DataCopyPad advisory."""
from __future__ import annotations

from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..query_utils import get_tool_args
from ..rules.dma_alignment_engine import analyze_dma_alignment
from ..reporting.advisory_report import advisory_display_string


def dma_alignment_engine_node(state: GeneratorAgentState) -> Dict[str, Any]:
    query = state.get("current_query", "") or ""
    args = get_tool_args(state)
    result = analyze_dma_alignment(query, args)
    display_text = advisory_display_string(result)
    round_num = state.get("query_round_count", 0) + 1
    log_entry = {
        "round": round_num,
        "tool": "dma_alignment_engine",
        "query": query or "[dma_alignment_engine]",
        "response": display_text,
    }
    print(f"[Round {round_num}] 工具=dma_alignment_engine")
    return {
        "dma_alignment_engine_results": [display_text],
        "dma_alignment_engine_result": result,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
