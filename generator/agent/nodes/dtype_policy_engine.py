"""dtype_policy_engine built-in: accumulation / cast advisory (no doc retrieval)."""
from __future__ import annotations

from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..query_utils import get_tool_args
from ..rules.dtype_policy_engine import analyze_dtype_policy
from ..reporting.advisory_report import advisory_display_string


def dtype_policy_engine_node(state: GeneratorAgentState) -> Dict[str, Any]:
    query = state.get("current_query", "") or ""
    args = get_tool_args(state)
    result = analyze_dtype_policy(query, args)
    display_text = advisory_display_string(result)
    round_num = state.get("query_round_count", 0) + 1
    log_entry = {
        "round": round_num,
        "tool": "dtype_policy_engine",
        "query": query or "[dtype_policy_engine]",
        "response": display_text,
    }
    print(f"[Round {round_num}] 工具=dtype_policy_engine")
    return {
        "dtype_policy_engine_results": [display_text],
        "dtype_policy_engine_result": result,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
