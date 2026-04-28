"""
Tool selection node for generator agent.

LLM outputs a single JSON object (ToolChoiceV1) selecting the next action.
The object may include an optional compact ``thinking`` summary for reporting.
"""
from __future__ import annotations

import time
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from generator.repo_root import REPO_ROOT
from ..agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from ..agent_config import (
    AgentToolMode,
    NO_TOOL,
    normalize_tool_choice_name,
)
from ..query_utils import extract_api_name, extract_chip_name, extract_npu_query_params
from ..tool_choice import ToolChoiceV1, parse_tool_choice_json
from ..tool_registry import get_tool_registry


_CODE_SEARCH_SOURCES = {"all", "cann_skills", "asc_devkit"}
_CODE_SEARCH_ARTIFACT_TYPES = {"kernel", "host", "tiling", "opdef", "pybind"}
_CODE_SEARCH_OPERATOR_FAMILIES = {"activation", "convolution", "matrix", "normalization", "elementwise", "pooling", "reduce"}
_CODE_SEARCH_SOURCE_GROUPS = {
    "asc_devkit_custom_op_host",
    "asc_devkit_custom_op_kernel",
    "asc_devkit_libraries_activation",
    "asc_devkit_libraries_matrix",
    "asc_devkit_basic_api_compute",
}


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
    base = user_msgs[0].content if user_msgs else state["messages"][-1].content
    ref_block = _load_pytorch_reference_block(state)
    if not ref_block:
        return base
    return f"{base}\n\n{ref_block}"


def _load_pytorch_reference_block(state: GeneratorAgentState) -> str:
    """Load per-op PyTorch reference source and append it to task specification."""
    op = (state.get("op_name") or "").strip()
    category = (state.get("category") or "").strip()
    if not op or not category:
        return ""
    ref_path = Path(REPO_ROOT) / "vendor" / "mkb" / "reference" / category / f"{op}.py"
    if not ref_path.is_file():
        return ""
    try:
        ref_src = ref_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    ref_src = ref_src.strip()
    if not ref_src:
        return ""
    max_chars = 6000
    if len(ref_src) > max_chars:
        ref_src = ref_src[:max_chars] + "\n# ... truncated ..."
    return (
        f"PyTorch reference implementation ({ref_path.as_posix()}):\n"
        f"```python\n{ref_src}\n```"
    )


def _summarize_existing_results(state: GeneratorAgentState) -> str:
    kb_results = state.get("kb_results", [])
    web_results = state.get("web_results", [])
    code_rag_results = state.get("code_rag_results", [])
    code_search_snippet_results = state.get("code_search_snippet_results", [])
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
    ascend_search_results = state.get("ascend_search_results", [])
    ascend_fetch_results = state.get("ascend_fetch_results", [])
    ascend_allowed_urls = state.get("ascend_search_allowed_urls", [])
    reg_results = state.get("registered_tool_results", [])

    existing = ""
    if kb_results:
        existing += "KB results (excerpt):\n" + "\n".join(kb_results[:2]) + "\n\n"
    if web_results:
        existing += "Web results (excerpt):\n" + "\n".join(web_results[:2]) + "\n\n"
    if code_rag_results:
        existing += "Code RAG results (excerpt):\n" + "\n".join(code_rag_results[:1]) + "\n\n"
    if code_search_snippet_results:
        existing += (
            "Code Search Snippect results (excerpt):\n"
            + "\n".join(code_search_snippet_results[:1])
            + "\n\n"
        )
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
    if ascend_search_results:
        existing += "Ascend docs search (excerpt):\n" + "\n".join(ascend_search_results[:2]) + "\n\n"
    if ascend_fetch_results:
        existing += "Ascend docs fetch (excerpt):\n" + "\n".join(ascend_fetch_results[:2]) + "\n\n"
    if ascend_allowed_urls:
        existing += (
            "Ascend fetch allowed URLs (excerpt):\n"
            + "\n".join(str(x) for x in ascend_allowed_urls[:5])
            + "\n\n"
        )
    if reg_results:
        existing += "Registered tool results (excerpt):\n" + "\n".join(reg_results[:2]) + "\n\n"
    return existing


def _summarize_called_tools(state: GeneratorAgentState) -> str:
    """Return a compact list of tool keys already called this session (from tool_calls_log)."""
    logs = state.get("tool_calls_log") or []
    called: List[str] = []
    for entry in logs:
        tool = (entry.get("tool") or "").strip()
        if tool and tool not in called:
            called.append(tool)
    return ", ".join(called) if called else ""


def _normalize_freeform_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _canonical_api_name(api_name: str) -> str:
    value = str(api_name or "").strip()
    for prefix in ("AscendC::", "ascendc::"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _stable_choice_signature(tool_name: str, query: str, args: Dict[str, Any] | None) -> str:
    normalized_args = dict(args or {})
    if tool_name in {"env_check_api", "api_lookup", "api_constraint", "api_alternative"}:
        api_name = _canonical_api_name(extract_api_name(query, args=normalized_args))
        normalized_args["api_name"] = api_name
        if tool_name in {"env_check_api", "api_lookup", "api_alternative"}:
            payload = {
                "tool": tool_name,
                "api_name": api_name,
                "args": {"api_name": api_name},
            }
            return json.dumps(payload, sort_keys=True, ensure_ascii=False)
        normalized_args = {
            key: value
            for key, value in normalized_args.items()
            if value is not None and value != ""
        }
        payload = {
            "tool": tool_name,
            "api_name": api_name,
            "args": normalized_args,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload = {
        "tool": tool_name,
        "query": _normalize_freeform_query(query),
        "args": normalized_args,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


_RESULT_LIST_KEYS = {
    "kb": "kb_results",
    "web": "web_results",
    "code_rag": "code_rag_results",
    "code_search_snippet": "code_search_snippet_results",
    "env_check_env": "env_check_results",
    "env_check_npu": "env_check_results",
    "env_check_api": "env_check_results",
    "kb_shell_search": "kb_shell_search_results",
    "api_lookup": "api_lookup_results",
    "api_constraint": "api_constraint_results",
    "api_alternative": "api_alternative_results",
    "tiling_calc": "tiling_calc_results",
    "tiling_validate": "tiling_validate_results",
    "npu_arch": "npu_arch_results",
    "code_style": "code_style_results",
    "security_check": "security_check_results",
    "ascend_search": "ascend_search_results",
    "ascend_fetch": "ascend_fetch_results",
}


def _tool_has_nonempty_result(state: GeneratorAgentState, tool_name: str) -> bool:
    result_key = _RESULT_LIST_KEYS.get(tool_name)
    if not result_key:
        return False
    results = state.get(result_key) or []
    return any(str(item or "").strip() for item in results)


def _tool_choice_already_seen(
    state: GeneratorAgentState,
    tool_name: str,
    query: str,
    args: Dict[str, Any] | None,
) -> bool:
    target = _stable_choice_signature(tool_name, query, args)
    for entry in reversed(state.get("tool_calls_log") or []):
        prev_tool = str(entry.get("tool") or "").strip()
        if prev_tool != tool_name:
            continue
        prev_query = str(entry.get("query") or "")
        prev_args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        if _stable_choice_signature(prev_tool, prev_query, prev_args) == target:
            return True
    return False


def _registry_tools_for_prompt(tool_mode: AgentToolMode) -> List[Any]:
    reg = get_tool_registry()
    out: List[Any] = []
    for key in sorted(tool_mode):
        spec = reg.get(key)
        if spec is not None:
            out.append(spec)
    return out


def _format_enabled_tools_manual(specs: List[Any]) -> str:
    """Per-enabled-tool block: name, summary, parameters, usage_guidance, examples (for choose_tool prompt)."""
    blocks: List[str] = []
    for spec in sorted(specs, key=lambda s: getattr(s, "name", "")):
        name = getattr(spec, "name", "") or ""
        display = getattr(spec, "display_name", "") or ""
        desc = (getattr(spec, "description", None) or "").strip().rstrip(".")
        params = (getattr(spec, "parameter_docs", None) or "").strip()
        ug = (getattr(spec, "usage_guidance", None) or "").strip()
        examples = list(getattr(spec, "examples", None) or [])
        lines: List[str] = [
            f"### `{name}` — {display}",
            f"Summary: {desc}.",
            f"Parameters: {params}",
        ]
        if ug:
            lines.append("Usage guidance:")
            lines.append(ug)
        if examples:
            lines.append("Examples:")
            for ex in examples:
                lines.append(f"  - {ex}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _build_tool_selection_prompt(
    user_question: str,
    existing_results: str,
    tool_mode: AgentToolMode,
    round_count: int,
    called_tools_str: str = "",
) -> str:
    specs = _registry_tools_for_prompt(tool_mode)
    if not specs:
        return ""

    tools_manual = _format_enabled_tools_manual(specs)
    tool_keys_csv = ", ".join(f"`{s.name}`" for s in sorted(specs, key=lambda x: x.name))
    at_max = round_count >= MAX_QUERY_ROUNDS
    rounds_left = MAX_QUERY_ROUNDS - round_count

    called_note = (
        f"Tools already called this session: **{called_tools_str}**"
        if called_tools_str
        else "Tools already called this session: **(none yet)**"
    )

    routing_instructions = (
        "You are a **tool orchestrator**. Your only job is to pick the next tool to call. "
        "You are NOT the model that writes code — a separate agent does that after all tools finish.\n\n"
        f"**Rounds remaining: {rounds_left} of {MAX_QUERY_ROUNDS}.**\n"
        "You need to make **only the minimum tool calls needed** to gather evidence before the downstream "
        "code-generation agent can produce a correct Ascend C kernel. Stop calling tools once the retrieved "
        "evidence is already actionable for code generation.\n\n"
        "**You MUST call a tool** unless the \"Already retrieved\" section already contains "
        "comprehensive, task-specific evidence that covers API signatures, usage patterns, AND "
        "hardware constraints. If that section is empty, thin, or covers only one aspect, call a tool.\n\n"
        "Across rounds, combine different **enabled** information sources (docs, code, runtime checks, "
        "online references as available) until the downstream agent has enough complementary evidence.\n\n"
        f"{called_note}\n"
        f"**Do NOT repeat the same tool for the same API symbol or near-identical query** after it already returned non-empty evidence. "
        f"Repeat only when you are adding genuinely new structure (for example, `api_constraint` after `api_lookup`, or a narrower `kb_shell_search` after an empty broad search).\n\n"
        "If only one tool is enabled for this session and it already returned a non-empty, task-relevant result, prefer `ANSWER` instead of burning more rounds on paraphrased repeats.\n\n"
        "When `api_lookup` returns a concrete API and constraints still matter, prefer `api_constraint` next instead of asking `api_lookup` again.\n\n"
        "**ANSWER is a last resort.** Output ANSWER only when:\n"
        "  (a) the retrieved content already comprehensively covers the task, OR\n"
        "  (b) you have no rounds left.\n\n"
        "Before choosing a tool, think in this order: current goal, missing information, why the next tool "
        "is the best enabled choice, and what evidence you expect back. Put that brief audit trail into the "
        "optional `thinking` object using short factual strings. Do not put hidden chain-of-thought or long prose there.\n\n"
        "**Strict output rules:**\n"
        "- This step selects **exactly one** tool key (or ANSWER): output **one** JSON object only; "
        "never bundle multiple tools in one object.\n"
        "- Output **one** JSON object only. No markdown fences, no commentary before or after.\n"
        '- Tool call (wire format v1): `{"tool": "<key>", "query": "<question or exact symbol>", "args": null|{...}, '
        '"thinking": {"goal": "...", "missing_info": "...", "why_tool": "...", "expected_output": "..."}}` — '
        'the key name `query` is fixed by the parser for this protocol; semantics follow each tool\'s Parameters.\n'
        '- ANSWER: `{"tool": "ANSWER", "query": "", "args": null}`\n'
        "- Never put code, partial answers, or any narrative inside `args`.\n\n"
        "**Structured-args policy:**\n"
        "- For `env_check_npu`, set `args.query_type` when it is inferable; only include `args.device_id` when the request explicitly names a device/card.\n"
        "- For `env_check_api`, `api_lookup`, `api_constraint`, and `api_alternative`, include `args.api_name` whenever you can infer one exact API symbol.\n"
        "- For `api_constraint`, include any obvious structured context such as `count`, `dtype`, `repeat_times`, or `is_gm_to_ub` when it is explicitly present.\n"
        "- For `code_search_snippet`, set `args.source` to `all`, `cann_skills`, or `asc_devkit`; default to `all` when no narrower source is required.\n"
        "- For `code_search_snippet`, also include inferable `args.artifact_types`, `args.operator_families`, and `args.source_groups` when the query clearly targets host/kernel/tiling/opdef or activation/matrix/convolution examples.\n\n"
    )

    prompt = (
        f"Task specification:\n{user_question}\n\n"
        f"{routing_instructions}"
        f"Available tools for this session (`tool` must be one of: {tool_keys_csv}, or `ANSWER`):\n\n"
        f"{tools_manual}"
        "Tool-choice JSON (v1 — keys are fixed for parsing): object with string fields `tool` and `query`, "
        "and `args` as null or object. An optional `thinking` object may carry a short decision summary for reporting. "
        "Meaning of `query` vs `args` is **not** generic: follow each tool's **Parameters** and **Usage guidance** above.\n"
        "Additional hard requirements:\n"
        "- Emit exactly **one** JSON object; no markdown fences and no prose outside that object.\n"
        "- Follow each tool's **Parameters** and **Usage guidance** above for `query` / `args`; use "
        '`"args": null` when that tool does not require a structured object.\n'
        "- When possible, populate `thinking` with a concise summary of goal, missing info, tool rationale, and expected output.\n"
        '- `"tool"`: lowercase key from the enabled list above, or `ANSWER`.\n'
        '- `"query"`: string slot for the tool\'s primary text request in this protocol; use `""` when `tool` is `ANSWER`.\n'
        '- For **ANSWER** use exactly `{"tool":"ANSWER","query":"","args":null}` as JSON; never put generated '
        "code, `project_json_src`, or any narrative inside `args`.\n"
    )
    if existing_results:
        prompt += f"\nAlready retrieved:\n{existing_results}\n"
    if at_max:
        prompt += (
            f"\nMaximum rounds reached ({MAX_QUERY_ROUNDS}). "
            'Output exactly: {"tool":"ANSWER","query":"","args":null}\n'
        )
    prompt += "\nOutput JSON only:"
    return prompt


def _query_mentions_explicit_device(query: str) -> bool:
    text = (query or "").lower()
    return bool(
        re.search(r"(?:device|card|npu)\s*#?\s*\d+", text)
        or re.search(r"设备\s*\d+", text)
        or re.search(r"-i\s*\d+", text)
    )


def _normalize_string_list(values: Any, allowed: set[str]) -> List[str]:
    if isinstance(values, str):
        items = re.split(r"[,|]", values)
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        items = []
    out: List[str] = []
    seen = set()
    for item in items:
        normalized = str(item or "").strip().lower().replace("-", "_")
        if not normalized or normalized not in allowed or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _infer_code_search_artifact_types(query: str) -> List[str]:
    lowered = (query or "").lower()
    inferred: List[str] = []
    if any(token in lowered for token in ("infer shape", "infershape", "getinputshape", "setoutputshape", "op_host", "host")):
        inferred.append("host")
    if any(token in lowered for token in ("tiling", "setblockdim", "workspace", "tile")):
        inferred.append("tiling")
    if any(token in lowered for token in ("project_json", "opdef", "register op", "registration", "input_desc", "output_desc", "op_def_registry")):
        inferred.append("opdef")
    if any(token in lowered for token in ("pybind", "torch.library", "binding", "python bind")):
        inferred.append("pybind")
    if any(token in lowered for token in ("kernel", "gm_addr", "__aicore__", "tpipe", "tque", "localtensor", "globaltensor", "datacopy", "kernel_operator")):
        inferred.append("kernel")
    return _normalize_string_list(inferred, _CODE_SEARCH_ARTIFACT_TYPES)


def _infer_code_search_operator_families(query: str) -> List[str]:
    lowered = (query or "").lower()
    inferred: List[str] = []
    if any(token in lowered for token in ("gelu", "relu", "softmax", "activation", "leakyrelu")):
        inferred.append("activation")
    if any(token in lowered for token in ("conv", "convolution", "pointwise", "depthwise")):
        inferred.append("convolution")
    if any(token in lowered for token in ("matmul", "gemm", "mma", "gemv", "matrix")):
        inferred.append("matrix")
    if any(token in lowered for token in ("layernorm", "rmsnorm", "normalization")):
        inferred.append("normalization")
    if any(token in lowered for token in ("pool", "pooling", "maxpool", "avgpool")):
        inferred.append("pooling")
    if any(token in lowered for token in ("reduce", "reduction", "reducesum", "reducemax", "reducemin")):
        inferred.append("reduce")
    if any(token in lowered for token in ("elementwise", "broadcast", "add", "sub", "mul", "div")):
        inferred.append("elementwise")
    return _normalize_string_list(inferred, _CODE_SEARCH_OPERATOR_FAMILIES)


def _infer_code_search_source_groups(query: str, artifact_types: List[str], operator_families: List[str]) -> List[str]:
    inferred: List[str] = []
    lowered = (query or "").lower()
    if any(kind in artifact_types for kind in ("host", "tiling", "opdef")):
        inferred.append("asc_devkit_custom_op_host")
    if "kernel" in artifact_types:
        inferred.append("asc_devkit_custom_op_kernel")
    if "activation" in operator_families:
        inferred.append("asc_devkit_libraries_activation")
    if "matrix" in operator_families or "convolution" in operator_families:
        inferred.append("asc_devkit_libraries_matrix")
    if any(token in lowered for token in ("maxs", "mins", "muls", "add", "sub", "datacopy")):
        inferred.append("asc_devkit_basic_api_compute")
    return _normalize_string_list(inferred, _CODE_SEARCH_SOURCE_GROUPS)


def _normalize_choice_args(tool_name: str, query: str, args: Dict[str, Any] | None) -> Dict[str, Any] | None:
    current = dict(args or {})

    if tool_name == "code_search_snippet":
        source = str(current.get("source") or "asc_devkit").strip().lower()
        if source == "cann_skills":
            source = "asc_devkit"
        current["source"] = source if source in _CODE_SEARCH_SOURCES else "asc_devkit"
        singular_family = str(current.get("operator_family") or "").strip().lower().replace("-", "_")
        if singular_family and singular_family in _CODE_SEARCH_OPERATOR_FAMILIES:
            existing = current.get("operator_families")
            merged = []
            if isinstance(existing, (list, tuple, set)):
                merged.extend(str(item) for item in existing)
            elif isinstance(existing, str):
                merged.append(existing)
            merged.append(singular_family)
            current["operator_families"] = merged
        artifact_types = _normalize_string_list(current.get("artifact_types"), _CODE_SEARCH_ARTIFACT_TYPES)
        if not artifact_types:
            artifact_types = _infer_code_search_artifact_types(query)
        if artifact_types:
            current["artifact_types"] = artifact_types
        else:
            current.pop("artifact_types", None)

        operator_families = _normalize_string_list(current.get("operator_families"), _CODE_SEARCH_OPERATOR_FAMILIES)
        if not operator_families:
            operator_families = _infer_code_search_operator_families(query)
        if operator_families:
            current["operator_families"] = operator_families
        else:
            current.pop("operator_families", None)

        source_groups = _normalize_string_list(current.get("source_groups"), _CODE_SEARCH_SOURCE_GROUPS)
        if not source_groups:
            source_groups = _infer_code_search_source_groups(query, artifact_types, operator_families)
        if source_groups:
            current["source_groups"] = source_groups
        else:
            current.pop("source_groups", None)

        if current["source"] == "all" and source_groups and all(group.startswith("asc_devkit_") for group in source_groups):
            current["source"] = "asc_devkit"
        current.pop("operator_family", None)
        return current

    if tool_name == "npu_arch":
        chip_name = extract_chip_name(query, args=current)
        if chip_name:
            current["chip_name"] = chip_name
        return current or None

    if tool_name == "env_check_npu":
        query_type, device_id = extract_npu_query_params(query, args=current)
        current["query_type"] = query_type
        if query_type in {"info", "list"} and not _query_mentions_explicit_device(query):
            current.pop("device_id", None)
        elif device_id is not None:
            current["device_id"] = device_id
        else:
            current.pop("device_id", None)
        return current

    if tool_name in {"env_check_api", "api_lookup", "api_constraint", "api_alternative"}:
        api_name = extract_api_name(query, args=current)
        if api_name != "unknown":
            current["api_name"] = api_name
        else:
            current.pop("api_name", None)

        if tool_name == "api_constraint":
            lowered = query.lower()
            for key in ("count", "repeat_times", "ub_usage_bytes", "ub_capacity_bytes"):
                if key in current:
                    continue
                match = re.search(rf"{key}\s*[=:]\s*(\d+)", lowered, re.IGNORECASE)
                if match:
                    current[key] = int(match.group(1))
            for dtype in ("float", "half", "float16", "bfloat16", "int32", "int16", "int8"):
                if dtype in lowered and "dtype" not in current:
                    current["dtype"] = dtype
                    break
            if "is_gm_to_ub" not in current and "gm" in lowered and ("ub" in lowered or "global" in lowered):
                current["is_gm_to_ub"] = True

        if tool_name == "api_alternative":
            reason = current.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                current["reason"] = "not found"

        return current or None

    return current or None


def _resolve_json_choice(
    choice: ToolChoiceV1,
    tool_mode: AgentToolMode,
    user_question: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Map validated ToolChoiceV1 to next_action, current_query, and tool_choice_json dict."""
    canon = normalize_tool_choice_name(choice.tool.strip())
    q = (choice.query or "").strip() or user_question.strip()
    normalized_args = _normalize_choice_args(canon or choice.tool.strip(), q, choice.args)
    choice_json = {"tool": canon or choice.tool.strip(), "query": choice.query or "", "args": normalized_args}
    if choice.thinking is not None:
        choice_json["thinking"] = choice.thinking
    if canon == "answer":
        answer_json = {"tool": "ANSWER", "query": "", "args": None}
        if choice.thinking is not None:
            answer_json["thinking"] = choice.thinking
        return "ANSWER", "", answer_json
    if canon is None or canon not in tool_mode:
        raise RuntimeError("invalid tool for _resolve_json_choice (caller must validate)")
    choice_json["tool"] = canon
    return canon, q, choice_json


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


def _reasoning_log_entry(
    *,
    round_num: int,
    prompt: str,
    raw_model_output: str,
    reasoning_content: str,
    parsed_ok: bool,
    selected_tool: str,
    structured_thinking: Dict[str, str] | None = None,
    structured_args: Dict[str, Any] | None = None,
    parse_error: str = "",
) -> Dict[str, Any]:
    def _clip(text: str, max_len: int) -> str:
        value = text or ""
        return value if len(value) <= max_len else value[:max_len] + "...(truncated)"

    return {
        "ts": time.time(),
        "round": round_num,
        "parsed_ok": parsed_ok,
        "selected_tool": selected_tool,
        "parse_error": parse_error,
        "args": dict(structured_args or {}),
        "prompt_excerpt": _clip(prompt, 2000),
        "raw_model_output": _clip(raw_model_output, 4000),
        "reasoning_content": _clip(reasoning_content, 4000),
        "thinking": {
            str(key): _clip(str(value), 400)
            for key, value in (structured_thinking or {}).items()
        },
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

    called_tools_str = _summarize_called_tools(state)
    prompt = _build_tool_selection_prompt(user_question, existing_results, tool_mode, round_count, called_tools_str)
    
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

    raw_response, reasoning = _openai_completion(
        client, model, [{"role": "user", "content": prompt}], stream=False
    )
    choice, err = parse_tool_choice_json(raw_response)
    if choice is None:
        new_round = round_count + 1
        entry = _parse_error_payload(err or "unknown", raw_response, new_round)
        reasoning_entry = _reasoning_log_entry(
            round_num=new_round,
            prompt=prompt,
            raw_model_output=raw_response,
            reasoning_content=reasoning or "",
            parsed_ok=False,
            selected_tool="",
            structured_args=choice.args if choice is not None else None,
            parse_error=err or "unknown",
        )
        print(f"[choose_tool] JSON parse failed: {err!r}")
        return {
            **base_patch,
            "tool_choice_parse_failed": True,
            "query_round_count": new_round,
            "tool_choice_error_log": [entry],
            "tool_choice_reasoning_log": [reasoning_entry],
            "next_action": "",
            "current_query": "",
            "tool_choice_json": {},
        }

    canon = normalize_tool_choice_name(choice.tool)
    if canon != "answer" and canon is not None and canon not in tool_mode:
        new_round = round_count + 1
        msg = f"tool {choice.tool!r} resolved to {canon!r} which is not enabled in this session"
        entry = _semantic_tool_error_payload(choice, msg, raw_response, new_round)
        reasoning_entry = _reasoning_log_entry(
            round_num=new_round,
            prompt=prompt,
            raw_model_output=raw_response,
            reasoning_content=reasoning or "",
            parsed_ok=False,
            selected_tool=choice.tool,
            structured_thinking=choice.thinking,
            structured_args=choice.args,
            parse_error=msg,
        )
        print(f"[choose_tool] {msg}")
        return {
            **base_patch,
            "tool_choice_parse_failed": True,
            "query_round_count": new_round,
            "tool_choice_error_log": [entry],
            "tool_choice_reasoning_log": [reasoning_entry],
            "next_action": "",
            "current_query": "",
            "tool_choice_json": {},
        }

    next_action, current_query, tjson = _resolve_json_choice(choice, tool_mode, user_question)
    normalized_args = tjson.get("args") if isinstance(tjson.get("args"), dict) else None
    enabled_tools = list(tool_mode)
    repeated_choice = next_action != "ANSWER" and _tool_choice_already_seen(state, next_action, current_query, normalized_args)
    single_tool_repeat = (
        next_action != "ANSWER"
        and len(enabled_tools) == 1
        and enabled_tools[0] == next_action
        and _tool_has_nonempty_result(state, next_action)
    )
    if next_action != "ANSWER" and (repeated_choice or single_tool_repeat):
        api_name = extract_api_name(current_query, args=normalized_args)
        if (
            next_action == "api_lookup"
            and "api_constraint" in tool_mode
            and api_name != "unknown"
            and not _tool_has_nonempty_result(state, "api_constraint")
        ):
            next_action = "api_constraint"
            current_query = f"check constraints for {api_name}"
            tjson = {
                "tool": "api_constraint",
                "query": current_query,
                "args": {"api_name": api_name},
            }
        else:
            next_action = "ANSWER"
            current_query = ""
            tjson = {"tool": "ANSWER", "query": "", "args": None}
    reasoning_entry = _reasoning_log_entry(
        round_num=round_count + 1,
        prompt=prompt,
        raw_model_output=raw_response,
        reasoning_content=reasoning or "",
        parsed_ok=True,
        selected_tool=next_action,
        structured_thinking=choice.thinking,
        structured_args=tjson.get("args") if isinstance(tjson.get("args"), dict) else None,
    )
    print(f"[choose_tool] model reply: {raw_response[:120]!r} -> next_action={next_action}")
    return {
        **base_patch,
        "next_action": next_action,
        "current_query": current_query,
        "tool_choice_json": tjson,
        "tool_choice_reasoning_log": [reasoning_entry],
    }
