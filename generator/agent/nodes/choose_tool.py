"""
Tool selection node for generator agent.

LLM outputs a single JSON object (ToolChoiceV1) selecting the next action.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from ..agent_config import (
    AgentToolMode,
    NO_TOOL,
    normalize_tool_choice_name,
)
from ..tool_choice import ToolChoiceV1, parse_tool_choice_json
from ..tool_registry import get_tool_registry


def _openai_completion(
    client,
    model: str,
    messages: list,
    stream: bool = False,
) -> Tuple[str, str]:
    if not stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            or ""
        )
        if reasoning and not isinstance(reasoning, str):
            reasoning = ""
        return content, (reasoning if isinstance(reasoning, str) else "")

    stream_resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    content_parts: list = []
    reasoning_parts: list = []
    for chunk in stream_resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        r = (
            getattr(delta, "reasoning_content", None)
            or (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
        )
        if r:
            reasoning_parts.append(r)
    return "".join(content_parts).strip(), "".join(reasoning_parts)


def _extract_user_question(state: GeneratorAgentState) -> str:
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return user_msgs[0].content if user_msgs else state["messages"][-1].content


def _summarize_existing_results(state: GeneratorAgentState) -> str:
    kb_results = state.get("kb_results", [])
    web_results = state.get("web_results", [])
    code_rag_results = state.get("code_rag_results", [])
    env_check_results = state.get("env_check_results", [])
    kb_shell_results = state.get("kb_shell_search_results", [])
    api_lookup_results = state.get("api_lookup_results", [])
    api_constraint_results = state.get("api_constraint_results", [])
    api_alternative_results = state.get("api_alternative_results", [])
    tiling_calc_results = state.get("tiling_calc_results", [])
    tiling_validate_results = state.get("tiling_validate_results", [])
    npu_arch_results = state.get("npu_arch_results", [])
    code_style_results = state.get("code_style_results", [])
    security_check_results = state.get("security_check_results", [])
    reg_results = state.get("registered_tool_results", [])

    existing = ""
    if kb_results:
        existing += "KB results (excerpt):\n" + "\n".join(kb_results[:2]) + "\n\n"
    if web_results:
        existing += "Web results (excerpt):\n" + "\n".join(web_results[:2]) + "\n\n"
    if code_rag_results:
        existing += "Code RAG results (excerpt):\n" + "\n".join(code_rag_results[:1]) + "\n\n"
    if env_check_results:
        existing += "Environment check (excerpt):\n" + "\n".join(env_check_results[:1]) + "\n\n"
    if kb_shell_results:
        existing += "KB shell search (excerpt):\n" + "\n".join(kb_shell_results[:1]) + "\n\n"
    if api_lookup_results:
        existing += "API lookup (excerpt):\n" + "\n".join(api_lookup_results[:1]) + "\n\n"
    if api_constraint_results:
        existing += "API constraint (excerpt):\n" + "\n".join(api_constraint_results[:1]) + "\n\n"
    if api_alternative_results:
        existing += "API alternative (excerpt):\n" + "\n".join(api_alternative_results[:1]) + "\n\n"
    if tiling_calc_results:
        existing += "Tiling calc (excerpt):\n" + "\n".join(tiling_calc_results[:1]) + "\n\n"
    if tiling_validate_results:
        existing += "Tiling validate (excerpt):\n" + "\n".join(tiling_validate_results[:1]) + "\n\n"
    if npu_arch_results:
        existing += "NPU arch (excerpt):\n" + "\n".join(npu_arch_results[:1]) + "\n\n"
    if code_style_results:
        existing += "Code style (excerpt):\n" + "\n".join(code_style_results[:1]) + "\n\n"
    if security_check_results:
        existing += "Security check (excerpt):\n" + "\n".join(security_check_results[:1]) + "\n\n"
    if reg_results:
        existing += "Registered tool results (excerpt):\n" + "\n".join(reg_results[:2]) + "\n\n"
    return existing


def _registry_tools_for_prompt(tool_mode: AgentToolMode) -> List[Any]:
    reg = get_tool_registry()
    out: List[Any] = []
    for key in sorted(tool_mode):
        spec = reg.get(key)
        if spec is not None:
            out.append(spec)
    return out


def _build_tool_selection_prompt(
    user_question: str,
    existing_results: str,
    tool_mode: AgentToolMode,
    round_count: int,
) -> str:
    specs = _registry_tools_for_prompt(tool_mode)
    tools_desc: List[str] = []
    for spec in specs:
        desc = (spec.description or "").rstrip(".")
        tools_desc.append(
            f"{spec.name} ({spec.display_name}): {desc}. Parameters: {spec.parameter_docs}"
        )
    if not tools_desc:
        return ""

    tools_line = "\n".join(f"- {t}" for t in tools_desc)
    at_max = round_count >= MAX_QUERY_ROUNDS

    examples: List[str] = []
    example_idx = 1
    for spec in specs:
        for ex in (spec.examples or [])[:2]:
            examples.append(f"Example {example_idx} (tool={spec.name}):\n{ex}")
            example_idx += 1
    examples.append(
        'Example (downstream may proceed without further retrieval — use null args only):\n'
        '{"tool":"ANSWER","query":"","args":null}'
    )
    few_shot = "\n\n".join(examples)

    routing_instructions = (
        "Your role here is **orchestration only**. You are **not** the model that writes the Ascend C "
        "kernel or the six-string bundle. Another agent will read your JSON and either run a tool or "
        "produce the final code.\n\n"
        "**CANN / API context:** CANN releases and recommended usage patterns change quickly. Even when "
        "the task looks familiar, **prefer using the enabled tools** so the downstream agent gets the "
        "**latest** documentation and the **most canonical** Ascend C patterns. The tools available "
        "in this session are **high-signal and very effective** at retrieving authoritative material; "
        "lean on them rather than relying on static intuition alone.\n\n"
        "**What you must decide:**\n"
        "- Output **ANSWER** only when the task text **plus** the \"Already retrieved\" excerpts already "
        "give the downstream agent enough **fresh, task-specific** evidence (not generic memory). If "
        "that section is empty or thin, you should normally **call a tool first**.\n"
        "- Otherwise choose **one** tool and a focused English `query`. Prefer a tool whenever there is "
        "any doubt: retrieved material improves grounding, aligns with current CANN guidance, and "
        "reduces hallucination risk for the downstream model. Treat proactive retrieval as your "
        "responsibility to the pipeline.\n\n"
        "**Strict output rules (avoid common failures):**\n"
        "- Output **one** JSON object only. No markdown fences, no commentary before or after the object, "
        "and no second JSON or trailing code.\n"
        "- For **ANSWER**, you must use exactly: "
        '`{"tool":"ANSWER","query":"","args":null}`. '
        "**Never** put generated code, `project_json_src`, partial answers, or any narrative inside "
        '`args`. Putting the solution in `args` breaks parsing (\"Extra data\" / invalid JSON) and '
        "skips real tool use.\n"
        '- For a normal tool call, use `"args": null` unless you are using a **plugin** tool whose docs '
        "explicitly require a small parameter object.\n"
        '- `"query"` is the natural-language request for the tool; use `""` when `tool` is ANSWER.\n\n'
    )

    prompt = (
        f"Task specification (shared with the downstream code-generation agent):\n{user_question}\n\n"
        f"{routing_instructions}"
        f"Available tools for this session (field \"tool\" must be one of these keys or ANSWER):\n{tools_line}\n"
        "You must output exactly one JSON object.\n"
        "JSON fields:\n"
        '  "tool": string — lowercase tool key from the list above, or ANSWER;\n'
        '  "query": string — English request for that tool; empty string when tool is ANSWER;\n'
        '  "args": null or a small object only when a plugin documents structured parameters.\n'
    )
    if existing_results:
        prompt += f"\nAlready retrieved (may be empty):\n{existing_results}\n"
    if at_max:
        prompt += (
            f"\nYou have reached the maximum number of query rounds ({MAX_QUERY_ROUNDS}). "
            'You must output exactly: {"tool":"ANSWER","query":"","args":null}\n'
        )
    prompt += f"\nFormat reference:\n{few_shot}\n\n"
    prompt += "Output JSON only:"
    return prompt


def _resolve_json_choice(
    choice: ToolChoiceV1,
    tool_mode: AgentToolMode,
    user_question: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Map validated ToolChoiceV1 to next_action, current_query, and tool_choice_json dict."""
    canon = normalize_tool_choice_name(choice.tool.strip())
    q = (choice.query or "").strip() or user_question.strip()
    if canon == "answer":
        return "ANSWER", "", {"tool": "ANSWER", "query": "", "args": None}
    if canon is None or canon not in tool_mode:
        raise RuntimeError("invalid tool for _resolve_json_choice (caller must validate)")
    return canon, q, {"tool": canon, "query": choice.query or "", "args": choice.args}


def _semantic_tool_error_payload(
    choice: ToolChoiceV1,
    message: str,
    raw_model_output: str,
    round_after_burn: int,
) -> Dict[str, Any]:
    return {
        "kind": "tool_choice_semantic_error",
        "ts": time.time(),
        "round": round_after_burn,
        "error": message,
        "parsed_tool_field": choice.tool,
        "raw_model_output": (raw_model_output or "")[:8000],
    }


def _parse_error_payload(err: str, raw_model_output: str, round_after_burn: int) -> Dict[str, Any]:
    return {
        "kind": "tool_choice_parse_error",
        "ts": time.time(),
        "round": round_after_burn,
        "error": err or "unknown",
        "raw_model_output": (raw_model_output or "")[:8000],
    }


def choose_tool_node(
    state: GeneratorAgentState,
    client,
    model: str,
    tool_mode: AgentToolMode,
) -> Dict[str, Any]:
    user_question = _extract_user_question(state)
    existing_results = _summarize_existing_results(state)
    round_count = state.get("query_round_count", 0)

    base_patch: Dict[str, Any] = {}
    if state.get("tool_choice_parse_failed"):
        base_patch["tool_choice_parse_failed"] = False

    if tool_mode == NO_TOOL:
        return {
            **base_patch,
            "next_action": "ANSWER",
            "current_query": "",
            "tool_choice_json": {"tool": "ANSWER", "query": "", "args": None},
        }

    prompt = _build_tool_selection_prompt(user_question, existing_results, tool_mode, round_count)
    if not prompt:
        return {
            **base_patch,
            "next_action": "ANSWER",
            "current_query": "",
            "tool_choice_json": {"tool": "ANSWER", "query": "", "args": None},
        }

    at_max = round_count >= MAX_QUERY_ROUNDS
    if at_max:
        return {
            **base_patch,
            "next_action": "ANSWER",
            "current_query": "",
            "tool_choice_json": {"tool": "ANSWER", "query": "", "args": None},
        }

    raw_response, _ = _openai_completion(
        client, model, [{"role": "user", "content": prompt}], stream=False
    )
    choice, err = parse_tool_choice_json(raw_response)
    if choice is None:
        new_round = round_count + 1
        entry = _parse_error_payload(err or "unknown", raw_response, new_round)
        print(f"[choose_tool] JSON parse failed: {err!r}")
        return {
            **base_patch,
            "tool_choice_parse_failed": True,
            "query_round_count": new_round,
            "tool_choice_error_log": [entry],
            "next_action": "",
            "current_query": "",
            "tool_choice_json": {},
        }

    canon = normalize_tool_choice_name(choice.tool)
    if canon != "answer" and canon is not None and canon not in tool_mode:
        new_round = round_count + 1
        msg = f"tool {choice.tool!r} resolved to {canon!r} which is not enabled in this session"
        entry = _semantic_tool_error_payload(choice, msg, raw_response, new_round)
        print(f"[choose_tool] {msg}")
        return {
            **base_patch,
            "tool_choice_parse_failed": True,
            "query_round_count": new_round,
            "tool_choice_error_log": [entry],
            "next_action": "",
            "current_query": "",
            "tool_choice_json": {},
        }

    next_action, current_query, tjson = _resolve_json_choice(choice, tool_mode, user_question)
    print(f"[choose_tool] model reply: {raw_response[:120]!r} -> next_action={next_action}")
    return {
        **base_patch,
        "next_action": next_action,
        "current_query": current_query,
        "tool_choice_json": tjson,
    }
