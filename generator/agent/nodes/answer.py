"""
Answer node for generator agent.

Generate final kernel code based on all retrieved information.
"""
from typing import Dict, Any, Tuple

from langchain_core.messages import HumanMessage, AIMessage

from ..agent_state import GeneratorAgentState


def _openai_completion_stream(
    client,
    model: str,
    messages: list,
) -> Tuple[str, str]:
    """
    Call OpenAI-compatible API with streaming.

    Args:
        client: OpenAI client
        model: Model name
        messages: Message list

    Returns:
        Tuple of (content, reasoning)
    """
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


def _format_retrieved_content(state: GeneratorAgentState) -> str:
    """Format all retrieved content for the prompt (aligned with choose_tool summaries)."""
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
    registered_tool_results = state.get("registered_tool_results", [])

    parts: list = []

    if kb_results:
        kb_text = "\n\n".join(kb_results[:6])
        parts.append(f"[KB retrieval — Ascend C API docs]\n{kb_text}")

    if web_results:
        web_text = "\n\n".join(web_results[:6])
        parts.append(f"[Web search results]\n{web_text}")

    if code_rag_results:
        code_text = "\n\n".join(code_rag_results[:4])
        parts.append(f"[Code RAG retrieval]\n{code_text}")

    if env_check_results:
        env_text = "\n\n".join(env_check_results[:4])
        parts.append(f"[Environment checks]\n{env_text}")

    if kb_shell_results:
        parts.append("[KB shell search]\n" + "\n\n".join(kb_shell_results[:4]))

    if api_lookup_results:
        parts.append("[API signature lookup]\n" + "\n\n".join(api_lookup_results[:4]))

    if api_constraint_results:
        parts.append("[API constraint check]\n" + "\n\n".join(api_constraint_results[:4]))

    if api_alternative_results:
        parts.append("[API alternatives]\n" + "\n\n".join(api_alternative_results[:4]))

    if tiling_calc_results:
        parts.append("[Tiling calculation]\n" + "\n\n".join(tiling_calc_results[:4]))

    if tiling_validate_results:
        parts.append("[Tiling validation]\n" + "\n\n".join(tiling_validate_results[:4]))

    if npu_arch_results:
        parts.append("[NPU architecture]\n" + "\n\n".join(npu_arch_results[:4]))

    if code_style_results:
        parts.append("[Code style]\n" + "\n\n".join(code_style_results[:4]))

    if security_check_results:
        parts.append("[Security scan]\n" + "\n\n".join(security_check_results[:4]))

    if registered_tool_results:
        parts.append("[Registered tools]\n" + "\n\n".join(registered_tool_results[:6]))

    return "\n\n".join(parts) if parts else ""


def answer_node(
    state: GeneratorAgentState,
    client,
    model: str,
) -> Dict[str, Any]:
    """
    Answer node: generate final kernel code.

    Args:
        state: Current agent state
        client: OpenAI client
        model: Model name

    Returns:
        Dict with messages (AIMessage) and reasoning_content
    """
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else ""

    # Get base prompt if available
    base_prompt = state.get("base_prompt", user_question)

    # Format retrieved content
    ref_text = _format_retrieved_content(state)

    # Build system prompt
    if ref_text:
        system_prompt = (
            "You are an expert Ascend C kernel engineer. The following blocks are retrieval results "
            "and tool outputs gathered for the user task.\n"
            "Use them faithfully and produce the full Ascend C kernel / host / project artifacts "
            "the user asked for.\n\n"
            f"Retrieved context:\n{ref_text}\n\n"
            f"Original user instruction:\n{base_prompt}\n\n"
            "Generate the complete solution as instructed (code only where the task demands code)."
        )
    else:
        system_prompt = (
            "You are an expert Ascend C kernel engineer.\n"
            "Generate the complete Ascend C artifacts requested below.\n\n"
            f"User instruction:\n{base_prompt}"
        )

    # Call LLM with streaming
    content, reasoning = _openai_completion_stream(
        client, model,
        [{"role": "system", "content": system_prompt}]
    )

    print(f"[answer] done, output length={len(content)} chars")

    return {
        "messages": [AIMessage(content=content)],
        "reasoning_content": reasoning or "",
    }