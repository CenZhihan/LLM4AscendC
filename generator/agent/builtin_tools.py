"""
Register built-in agent tools into the process ToolRegistry for the current graph.

``build_agent_app`` clears the registry, registers one ``RegisteredToolSpec`` per
enabled built-in key in ``tool_mode``, then restores any user-registered plugin specs.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .agent_config import AgentToolMode, BUILTIN_TOOL_NAMES
from .tool_registry import RegisteredToolSpec, get_tool_registry
from .nodes import (
    api_alternative_node,
    api_constraint_node,
    api_lookup_node,
    code_rag_node,
    code_style_node,
    env_check_api_node,
    env_check_env_node,
    env_check_npu_node,
    kb_query_node,
    kb_shell_search_node,
    ascend_search_node,
    ascend_fetch_node,
    npu_arch_node,
    security_check_node,
    tiling_calc_node,
    tiling_validate_node,
    web_search_node,
)

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


def _meta() -> Dict[str, Dict[str, Any]]:
    return {
        "kb": {
            "display_name": "Knowledge base",
            "description": "Query Huawei Ascend C API documentation (Chroma / LlamaIndex)",
            "parameter_docs": 'Use "query" for the English search question.',
            "examples": ['{"tool":"kb","query":"Ascend C GELU kernel API","args":null}'],
            "usage_guidance": (
                "Use **English** in `query` for best retrieval. Ask for concrete APIs, constraints, or "
                "patterns (e.g. elementwise tiling, DataCopy alignment, dtype rules)."
            ),
        },
        "web": {
            "display_name": "Web search",
            "description": "Search the public web for docs and tutorials (ddgs).",
            "parameter_docs": 'Use "query" for the search question.',
            "examples": ['{"tool":"web","query":"Ascend C custom operator tutorial","args":null}'],
            "usage_guidance": (
                "Put one focused search line in `query`; avoid a single overly broad word unless you "
                "are exploring."
            ),
        },
        "code_rag": {
            "display_name": "Code RAG",
            "description": "Retrieve similar Ascend C kernel implementations from the indexed code corpus.",
            "parameter_docs": 'Use "query" for what to look up in code.',
            "examples": ['{"tool":"code_rag","query":"Ascend C softmax kernel example","args":null}'],
            "usage_guidance": (
                "Describe the kernel pattern, op family, or API usage you need (shapes, fusion, tiling) "
                "in `query` in English."
            ),
        },
        "env_check_env": {
            "display_name": "Environment (CANN)",
            "description": "Summarize CANN / toolchain environment and broad API compatibility hints.",
            "parameter_docs": 'Use "query" for what to verify in the environment.',
            "examples": ['{"tool":"env_check_env","query":"check CANN environment","args":null}'],
            "usage_guidance": (
                "State what to verify (CANN version, compiler paths, generic compatibility) in `query`; "
                "keep it task-relevant."
            ),
        },
        "env_check_npu": {
            "display_name": "NPU device",
            "description": "Query NPU device status and resource usage via the env checker.",
            "parameter_docs": 'Use "query" for the device query intent. Prefer args like {"query_type":"memory|temp|power|usages|list|info","device_id":0}.',
            "examples": ['{"tool":"env_check_npu","query":"NPU memory for device 0","args":{"query_type":"memory","device_id":0}}'],
            "usage_guidance": (
                "Prefer structured `args`: `{\"query_type\":\"memory|temp|power|usages|list|info\","
                "\"device_id\":0}`. Put a short natural-language intent in `query` that matches "
                "`query_type` and device."
            ),
        },
        "env_check_api": {
            "display_name": "API header check",
            "description": "Verify whether an Ascend C API symbol appears in installed headers.",
            "parameter_docs": 'Use an exact symbol name in query or args.api_name. Never use generic words like "signature" or "constraints" as the API name.',
            "examples": ['{"tool":"env_check_api","query":"check if AscendC::DataCopy exists","args":{"api_name":"AscendC::DataCopy"}}'],
            "usage_guidance": (
                "Identify **one concrete API symbol** first. Put the exact symbol in `query` or "
                "`args.api_name`. Good symbols: `AscendC::DataCopy`, `Muls`, `DataCopyPad`, `MatmulType`, "
                "`GlobalTensor::SetValue`. Bad placeholders as the API name: `api`, `signature`, "
                "`signatures`, `constraints`, `alternative`, `details`, `docs`."
            ),
        },
        "kb_shell_search": {
            "display_name": "KB shell search",
            "description": "Run grep/find style search over packaged knowledge-base markdown trees.",
            "parameter_docs": 'Use "query" for path / pattern / intent.',
            "examples": ['{"tool":"kb_shell_search","query":"search DataCopy in Knowledge/api/","args":null}'],
            "usage_guidance": (
                "Put path, filename glob, or grep-like pattern in `query`; anchor to the packaged KB tree "
                "when possible."
            ),
        },
        "api_lookup": {
            "display_name": "API signature lookup",
            "description": "Look up API signatures, dtypes, and repeatTimes limits from structured docs.",
            "parameter_docs": 'Use an exact API symbol in query or args.api_name, e.g. AscendC::DataCopy or MatmulType. Do not pass generic meta words like "signatures".',
            "examples": ['{"tool":"api_lookup","query":"AscendC::DataCopy","args":{"api_name":"AscendC::DataCopy"}}'],
            "usage_guidance": (
                "Identify **one concrete API symbol** first. Good symbols: `AscendC::DataCopy`, `Muls`, "
                "`DataCopyPad`, `MatmulType`, `GlobalTensor::SetValue`. Bad placeholders: `api`, `signature`, "
                "`signatures`, `constraints`, `alternative`, `details`, `docs`. If useful, put the exact "
                "symbol in `args`, e.g. `{\"api_name\":\"AscendC::DataCopy\"}`."
            ),
        },
        "api_constraint": {
            "display_name": "API constraint check",
            "description": "Check alignment, blockCount, and platform constraints for an API use site.",
            "parameter_docs": 'Provide the exact API symbol plus structured context in args, e.g. {"api_name":"DataCopyPad","count":512,"dtype":"half","is_gm_to_ub":true}.',
            "examples": [
                '{"tool":"api_constraint","query":"check DataCopyPad constraints","args":{"api_name":"DataCopyPad","count":512,"dtype":"half","is_gm_to_ub":true}}'
            ],
            "usage_guidance": (
                "Use the same **single-symbol discipline** as `api_lookup`. `args` must carry `api_name` "
                "plus the use-site context (count, dtype, GM/UB direction, etc.) per parameter_docs."
            ),
        },
        "api_alternative": {
            "display_name": "API alternatives",
            "description": "Suggest equivalent APIs when the primary symbol is unavailable.",
            "parameter_docs": 'Provide the exact unavailable API in query or args.api_name and optionally args.reason.',
            "examples": [
                '{"tool":"api_alternative","query":"alternative for GlobalTensor::SetValue","args":{"api_name":"GlobalTensor::SetValue","reason":"not found"}}'
            ],
            "usage_guidance": (
                "Name the **exact unavailable symbol** in `query` or `args.api_name`; optional `args.reason` "
                "helps narrow replacements."
            ),
        },
        "tiling_calc": {
            "display_name": "Tiling calculation",
            "description": "Propose tiling parameters from operator shape / element counts.",
            "parameter_docs": 'Use "query" with sizes and operator kind.',
            "examples": [
                '{"tool":"tiling_calc","query":"tiling for 1024 float elements elementwise","args":null}'
            ],
            "usage_guidance": (
                "Include element counts, dtypes, operator kind, and chip hint in `query` so tiling can be "
                "grounded."
            ),
        },
        "tiling_validate": {
            "display_name": "Tiling validation",
            "description": "Validate tiling parameters against UB capacity and hardware rules.",
            "parameter_docs": 'Use "query" with tiling JSON or parameters.',
            "examples": [
                '{"tool":"tiling_validate","query":"validate tiling chip=DAV_2201 block_num=4","args":null}'
            ],
            "usage_guidance": (
                "Paste or describe the proposed tiling parameters and target chip in `query` so rules can "
                "be checked."
            ),
        },
        "npu_arch": {
            "display_name": "NPU architecture",
            "description": "Return UB size, compile macros, and feature flags for a chip name.",
            "parameter_docs": 'Use "query" with chip id, e.g. Ascend910B2.',
            "examples": ['{"tool":"npu_arch","query":"Ascend910B2 chip specs","args":null}'],
            "usage_guidance": (
                "Put the **chip id** (e.g. Ascend910B2) in `query`; ask for UB, macros, or feature flags as "
                "needed."
            ),
        },
        "code_style": {
            "display_name": "Code style",
            "description": "Check Ascend C coding-style rules on kernel / host snippets.",
            "parameter_docs": 'Use "query" with code excerpt or file intent.',
            "examples": ['{"tool":"code_style","query":"check style for elementwise kernel","args":null}'],
            "usage_guidance": (
                "Embed or point to the smallest relevant **code excerpt** in `query` (or describe the "
                "snippet role) so checks are focused."
            ),
        },
        "security_check": {
            "display_name": "Security patterns",
            "description": "Scan for risky patterns (dynamic alloc, std:: misuse, etc.) in kernel code.",
            "parameter_docs": 'Use "query" with code excerpt.',
            "examples": [
                '{"tool":"security_check","query":"security scan elementwise kernel snippet","args":null}'
            ],
            "usage_guidance": (
                "Provide the **kernel/host snippet** or a tight description of what to scan in `query`."
            ),
        },
        "ascend_search": {
            "display_name": "Ascend docs search",
            "description": "Search Ascend online docs.",
            "parameter_docs": 'Use "query" in Chinese only (关键词必须中文).',
            "examples": ['{"tool":"ascend_search","query":"AscendC DataCopy 对齐约束","args":null}'],
            "usage_guidance": (
                "Put **Chinese keywords only** in `query`. `lang`, `doc_type`, and `version` are fixed "
                "server-side — **do not** invent them in `args`."
            ),
        },
        "ascend_fetch": {
            "display_name": "Ascend docs fetch",
            "description": "Fetch one URL from previous ascend_search results and extract main_content/code_examples.",
            "parameter_docs": 'Use "query" containing exactly one URL from ascend_search history. If multiple URLs are provided, only the first is used.',
            "examples": [
                '{"tool":"ascend_fetch","query":"https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/85RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0104.html","args":null}'
            ],
            "usage_guidance": (
                "Put **one** URL from a **prior `ascend_search` result** in `query`. If multiple URLs appear "
                "in `query`, only the **first** is fetched."
            ),
        },
    }


def _handler_for(
    name: str,
    client: Any,
    model: str,
    kb_retriever: Any,
    web_retriever: Any,
    code_retriever: Any,
    env_retriever: Any,
    npu_arch_retriever: Any,
    tiling_retriever: Any,
    api_retriever: Any,
    code_quality_retriever: Any,
    kb_shell_retriever: Any,
    ascend_search_retriever: Any,
    ascend_fetch_retriever: Any,
) -> ToolHandler:
    def h(state: Dict[str, Any]) -> Dict[str, Any]:
        if name == "kb":
            return kb_query_node(state, client, model, kb_retriever)
        if name == "web":
            return web_search_node(state, client, model, web_retriever)
        if name == "code_rag":
            return code_rag_node(state, code_retriever)
        if name == "env_check_env":
            return env_check_env_node(state, env_retriever)
        if name == "env_check_npu":
            return env_check_npu_node(state, env_retriever)
        if name == "env_check_api":
            return env_check_api_node(state, env_retriever)
        if name == "npu_arch":
            return npu_arch_node(state, npu_arch_retriever)
        if name == "tiling_calc":
            return tiling_calc_node(state, tiling_retriever)
        if name == "tiling_validate":
            return tiling_validate_node(state, tiling_retriever)
        if name == "api_lookup":
            return api_lookup_node(state, api_retriever)
        if name == "api_constraint":
            return api_constraint_node(state, api_retriever)
        if name == "api_alternative":
            return api_alternative_node(state, api_retriever)
        if name == "code_style":
            return code_style_node(state, code_quality_retriever)
        if name == "security_check":
            return security_check_node(state, code_quality_retriever)
        if name == "kb_shell_search":
            return kb_shell_search_node(state, kb_shell_retriever)
        if name == "ascend_search":
            return ascend_search_node(state, ascend_search_retriever)
        if name == "ascend_fetch":
            return ascend_fetch_node(state, ascend_fetch_retriever)
        raise KeyError(name)

    return h


def snapshot_plugin_specs(tool_mode: AgentToolMode) -> Dict[str, RegisteredToolSpec]:
    """Copy user-registered plugin specs for keys in ``tool_mode`` before ``registry.clear()``."""
    reg = get_tool_registry()
    out: Dict[str, RegisteredToolSpec] = {}
    for key in tool_mode:
        if key in BUILTIN_TOOL_NAMES:
            continue
        spec = reg.get(key)
        if spec is not None:
            out[key] = spec
    return out


def register_builtin_tools_for_mode(
    tool_mode: AgentToolMode,
    *,
    client: Any,
    model: str,
    kb_retriever: Any = None,
    web_retriever: Any = None,
    code_retriever: Any = None,
    env_retriever: Any = None,
    npu_arch_retriever: Any = None,
    tiling_retriever: Any = None,
    api_retriever: Any = None,
    code_quality_retriever: Any = None,
    kb_shell_retriever: Any = None,
    ascend_search_retriever: Any = None,
    ascend_fetch_retriever: Any = None,
    plugin_snapshot: Optional[Dict[str, RegisteredToolSpec]] = None,
) -> None:
    """
    Clear registry, register built-ins present in ``tool_mode``, then re-register plugins from snapshot.
    """
    reg = get_tool_registry()
    reg.clear()
    meta = _meta()
    for name in sorted(tool_mode):
        if name not in BUILTIN_TOOL_NAMES:
            continue
        m = meta.get(name)
        if not m:
            continue
        reg.register(
            RegisteredToolSpec(
                name=name,
                display_name=str(m["display_name"]),
                description=str(m["description"]),
                parameter_docs=str(m["parameter_docs"]),
                handler=_handler_for(
                    name,
                    client,
                    model,
                    kb_retriever,
                    web_retriever,
                    code_retriever,
                    env_retriever,
                    npu_arch_retriever,
                    tiling_retriever,
                    api_retriever,
                    code_quality_retriever,
                    kb_shell_retriever,
                    ascend_search_retriever,
                    ascend_fetch_retriever,
                ),
                examples=list(m.get("examples") or []),
                usage_guidance=str(m.get("usage_guidance") or ""),
            ),
            allow_builtin_name=True,
        )
    if plugin_snapshot:
        for key, spec in sorted(plugin_snapshot.items(), key=lambda kv: kv[0]):
            if key in tool_mode:
                reg.register(spec, allow_builtin_name=False)
