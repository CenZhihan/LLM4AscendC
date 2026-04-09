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
    """Format all retrieved content for the prompt."""
    kb_results = state.get("kb_results", [])
    web_results = state.get("web_results", [])
    code_rag_results = state.get("code_rag_results", [])

    parts: list = []

    if kb_results:
        kb_text = "\n\n".join(kb_results)
        parts.append(f"【知识库检索结果（Ascend C API 文档）】\n{kb_text}")

    if web_results:
        web_text = "\n\n".join(web_results)
        parts.append(f"【网页搜索结果】\n{web_text}")

    if code_rag_results:
        code_text = "\n\n".join(code_rag_results)
        parts.append(f"【代码检索结果】\n{code_text}")

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
            "你是一个 Ascend C 算子开发专家。下面是根据用户问题检索到的参考资料。\n"
            "请基于这些内容，按照用户的要求生成完整的 Ascend C Kernel 代码。\n\n"
            f"参考资料：\n{ref_text}\n\n"
            f"用户原始要求：\n{base_prompt}\n\n"
            "请生成符合要求的完整代码。"
        )
    else:
        system_prompt = (
            "你是一个 Ascend C 算子开发专家。\n"
            "请根据你的知识生成完整的 Ascend C Kernel 代码。\n\n"
            f"用户要求：\n{base_prompt}"
        )

    # Call LLM with streaming
    content, reasoning = _openai_completion_stream(
        client, model,
        [{"role": "system", "content": system_prompt}]
    )

    print(f"[answer] 生成完成，代码长度: {len(content)} 字符")

    return {
        "messages": [AIMessage(content=content)],
        "reasoning_content": reasoning or "",
    }