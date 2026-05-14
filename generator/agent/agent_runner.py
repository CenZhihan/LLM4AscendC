"""
Agent runner: high-level API for kernel generation with agent.

Provides generate_kernel_with_agent() function for easy integration.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage

from .agent_config import (
    AgentToolMode,
    NO_TOOL,
    parse_tool_mode,
    tool_mode_to_string,
    get_llm_config_compatible,
)
from .agent_state import GeneratorAgentState, create_initial_state
from .agent_builder import build_agent_app
from .retrievers import KBRetriever, WebRetriever, CodeRetriever


@dataclass
class KernelGenerationTask:
    """Task definition for kernel generation."""
    language: str           # Target language: "ascendc", "cuda", "triton"
    op: str                 # Operator name: "gelu", "softmax" (or ca6k_00055 for CUDA-Agent)
    strategy_name: str      # Prompt strategy (MKB: add_shot…; CUDA-Agent script: one_shot|none)
    category: str           # Operator category: "activation", "matmul"
    # When set, prompt is built from CUDA-Agent-Ops-6K row (fused task); op should be ca6k_XXXXX.
    cuda_agent_row: Optional[Dict[str, Any]] = None
    cuda_agent_row_index: Optional[int] = None


@dataclass
class AgentGenerationResult:
    """Result from agent-based generation."""
    op: str
    generated_code: str                # Final generated kernel code
    reasoning: Optional[str] = None    # LLM reasoning content
    tool_usage: Optional[List[Dict[str, Any]]] = None  # Tool call logs
    report: Optional[Dict[str, Any]] = None            # Full report


def _build_base_prompt(language: str, strategy_name: str, op: str) -> str:
    """
    Build base prompt using existing prompt_generators.

    Args:
        language: Target language
        strategy_name: Prompt strategy name
        op: Operator name

    Returns:
        Generated prompt string
    """
    from generator.prompt_generators.prompt_registry import PROMPT_REGISTRY
    import importlib

    # Ensure strategy is loaded
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        try:
            importlib.import_module(f"prompt_generators.{language}_{strategy_name}")
        except ImportError:
            pass

    if language in PROMPT_REGISTRY and strategy_name in PROMPT_REGISTRY[language]:
        strategy = PROMPT_REGISTRY[language][strategy_name]
        return strategy.generate(op)

    # Fallback: simple prompt
    return f"Write a {language} kernel implementation for the `{op}` operator."


def _extract_final_answer(final_state: Dict[str, Any]) -> str:
    """Extract final generated code from state."""
    messages = final_state.get("messages", [])
    if messages:
        last = messages[-1]
        return getattr(last, "content", "") or ""
    return ""


def _build_report(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Build detailed report from final state."""
    parse_errors = final_state.get("tool_choice_error_log", [])
    choice_reasoning = final_state.get("tool_choice_reasoning_log", [])
    tool_calls_log = final_state.get("tool_calls_log", [])

    def _summarize_parse_entry(e: Dict[str, Any]) -> Dict[str, Any]:
        raw = e.get("raw_model_output") or ""
        if len(raw) > 2000:
            raw = raw[:2000] + "...(truncated)"
        return {
            "kind": e.get("kind"),
            "round": e.get("round"),
            "error": e.get("error"),
            "parsed_tool_field": e.get("parsed_tool_field"),
            "raw_model_output": raw,
            "ts": e.get("ts"),
        }

    def _summarize_reasoning_entry(e: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = e.get("reasoning_content") or ""
        if len(reasoning) > 2000:
            reasoning = reasoning[:2000] + "...(truncated)"
        thinking = e.get("thinking") if isinstance(e.get("thinking"), dict) else {}
        return {
            "round": e.get("round"),
            "parsed_ok": e.get("parsed_ok"),
            "selected_tool": e.get("selected_tool", ""),
            "parse_error": e.get("parse_error", ""),
            "thinking": thinking,
            "reasoning_content": reasoning,
            "ts": e.get("ts"),
        }

    def _summarize_repair_memory_selection(sel: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(sel)
        raw = out.get("raw_model_output") or ""
        if len(raw) > 2000:
            out["raw_model_output"] = raw[:2000] + "...(truncated)"
        return out

    tool_calls = [
        {
            "round": t.get("round"),
            "tool": t.get("tool", ""),
            "query": t.get("query", ""),
            "response": t.get("response", "")[:500] + "..."
            if len(t.get("response", "")) > 500
            else t.get("response", ""),
        }
        for t in tool_calls_log
    ]
    reasoning_by_round = {
        int(x.get("round")): x
        for x in choice_reasoning
        if isinstance(x, dict) and isinstance(x.get("round"), int)
    }
    for item in tool_calls:
        r = item.get("round")
        if isinstance(r, int):
            item["tool_choice"] = reasoning_by_round.get(r, {})

    repair_sel = final_state.get("repair_memory_selection")
    report_body: Dict[str, Any] = {
        "attempt_id": int(final_state.get("attempt_id") or 1),
        "reasoning_content": final_state.get("reasoning_content", ""),
        "final_generation_reasoning_content": final_state.get("reasoning_content", ""),
        "answer": _extract_final_answer(final_state),
        "repair_memories_applied": (
            final_state.get("retrieved_repair_memories_applied")
            if isinstance(final_state.get("retrieved_repair_memories_applied"), list)
            else []
        ),
        "tool_selection_trace": [_summarize_reasoning_entry(x) for x in choice_reasoning],
        "tool_calls": tool_calls,
        "tool_choice_parse_errors": [_summarize_parse_entry(x) for x in parse_errors],
        "kb_results_count": len(final_state.get("kb_results", [])),
        "web_results_count": len(final_state.get("web_results", [])),
        "code_rag_results_count": len(final_state.get("code_rag_results", [])),
        "ascend_search_results_count": len(final_state.get("ascend_search_results", [])),
        "ascend_fetch_results_count": len(final_state.get("ascend_fetch_results", [])),
        "ascend_search_allowed_urls_count": len(final_state.get("ascend_search_allowed_urls", [])),
    }
    if isinstance(repair_sel, dict) and repair_sel:
        report_body["repair_memory_selection"] = _summarize_repair_memory_selection(repair_sel)
    return report_body


def generate_kernel_with_agent(
    task: KernelGenerationTask,
    tool_mode: Union[AgentToolMode, str] = NO_TOOL,
    retriever: Optional[CodeRetriever] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    attempt_id: int = 1,
    repair_error_logs_raw: str = "",
    previous_attempt_code: str = "",
    ascend_search_version_filter: Optional[str] = None,
    retrieved_repair_memories: str = "",
    retrieved_repair_memories_applied: Optional[List[Dict[str, Any]]] = None,
    repair_memory_selection: Optional[Dict[str, Any]] = None,
    eval_mode: str = "full",
) -> AgentGenerationResult:
    """
    Generate kernel code using the integrated agent with KB, WEB, and Code RAG.

    Args:
        task: KernelGenerationTask with language, op, strategy_name, category
        tool_mode: Tool mode (``frozenset`` of tool keys or string like ``\"kb_only\"``, ``\"kb,web\"``, ``\"all\"``)
        retriever: Optional pre-loaded CodeRetriever for Code RAG
        llm_config: Optional LLM config (api_key, base_url, model)
        ascend_search_version_filter: Optional substring for Ascend docs ``version`` field filtering;
            ``None`` or empty means no restriction.
        repair_memory_selection: Optional dict from repair-memory selection LLM (ids, rationale, parse);
            written into ``report["repair_memory_selection"]`` when provided.

    Returns:
        AgentGenerationResult with generated_code, reasoning, and tool_usage

    Example:
        task = KernelGenerationTask(
            language="ascendc",
            op="gelu",
            strategy_name="add_shot",
            category="activation"
        )
        result = generate_kernel_with_agent(task, "all")
        print(result.generated_code)
    """
    # Parse tool mode if string
    parsed_mode = parse_tool_mode(tool_mode) if isinstance(tool_mode, str) else tool_mode

    # 1. Build base prompt
    if task.cuda_agent_row is not None:
        from generator.prompt_generators.cuda_agent_fused import build_cuda_agent_fused_prompt

        base_prompt = build_cuda_agent_fused_prompt(
            task.strategy_name,
            task.cuda_agent_row,
            task.op,
        )
    else:
        base_prompt = _build_base_prompt(task.language, task.strategy_name, task.op)

    # 2. Build and invoke agent
    app = build_agent_app(
        tool_mode=parsed_mode,
        llm_config=llm_config,
        code_retriever=retriever,
        ascend_search_version_filter=ascend_search_version_filter,
    )

    # 3. Create initial state
    initial_state = create_initial_state(
        base_prompt=base_prompt,
        op_name=task.op,
        category=task.category,
        language=task.language,
        strategy_name=task.strategy_name,
        attempt_id=attempt_id,
        repair_error_logs_raw=repair_error_logs_raw,
        previous_attempt_code=previous_attempt_code,
        retrieved_repair_memories=retrieved_repair_memories or "",
        retrieved_repair_memories_applied=retrieved_repair_memories_applied,
        repair_memory_selection=repair_memory_selection,
        eval_mode=eval_mode or "full",
    )

    # 4. Invoke agent
    mode_str = tool_mode_to_string(parsed_mode)
    print(
        f"[INFO] Starting agent for op={task.op}, tool_mode={mode_str}, "
        f"attempt={attempt_id}"
    )
    final_state = app.invoke(initial_state)

    # 5. Extract results
    generated_code = _extract_final_answer(final_state)
    reasoning = final_state.get("reasoning_content")
    tool_calls = final_state.get("tool_calls_log", [])
    report = _build_report(final_state)

    return AgentGenerationResult(
        op=task.op,
        generated_code=generated_code,
        reasoning=reasoning if reasoning else None,
        tool_usage=tool_calls if tool_calls else None,
        report=report,
    )


# Convenience function for simple usage
def generate_ascendc_kernel(
    op: str,
    category: str = "activation",
    strategy: str = "add_shot",
    tool_mode: Union[str, AgentToolMode] = "no_tool",
) -> str:
    """
    Simple function to generate Ascend C kernel.

    Args:
        op: Operator name
        category: Operator category
        strategy: Prompt strategy
        tool_mode: Tool mode (string preset or comma list, or ``frozenset`` of tool keys)

    Returns:
        Generated kernel code
    """
    task = KernelGenerationTask(
        language="ascendc",
        op=op,
        strategy_name=strategy,
        category=category,
    )
    result = generate_kernel_with_agent(task, tool_mode)
    return result.generated_code