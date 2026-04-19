"""
Unified dispatch node: run the tool handler registered for ``next_action``.
"""
from typing import Any, Dict

from ..agent_state import GeneratorAgentState
from ..tool_registry import get_tool_registry


def tool_dispatch_node(state: GeneratorAgentState) -> Dict[str, Any]:
    """
    Look up ``next_action`` in the tool registry and invoke its handler.

    Handlers return the same partial-state dict contract as other agent nodes.
    """
    name = (state.get("next_action") or "").strip().lower()
    if not name:
        round_num = state.get("query_round_count", 0) + 1
        msg = "[tool_dispatch_error] empty next_action"
        log_entry = {"round": round_num, "tool": "", "query": "", "response": msg}
        return {
            "registered_tool_results": [msg],
            "query_round_count": round_num,
            "tool_calls_log": [log_entry],
        }
    spec = get_tool_registry().get(name)
    if spec is None:
        round_num = state.get("query_round_count", 0) + 1
        msg = f"[tool_dispatch_error] unknown or unregistered tool: {name!r}"
        log_entry = {
            "round": round_num,
            "tool": name,
            "query": state.get("current_query", ""),
            "response": msg,
        }
        return {
            "registered_tool_results": [msg],
            "query_round_count": round_num,
            "tool_calls_log": [log_entry],
        }
    return spec.handler(dict(state))


def registered_tool_dispatch_node(state: GeneratorAgentState) -> Dict[str, Any]:
    """Backward-compatible alias for :func:`tool_dispatch_node`."""
    return tool_dispatch_node(state)
