from __future__ import annotations

from typing import Any, Dict, List, Tuple

from openai import OpenAI


def chat_completion_stream_content_reasoning(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str]:
    """
    Same aggregation as ``answer._openai_completion_stream``:
    DeepSeek / xi-ai often put the **final answer** in streaming ``delta.content`` and
    chain-of-thought in ``delta.reasoning_content``. Non-streaming ``message.content`` may
    stay empty while reasoning is filled — selection must stream like the main agent.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        r = getattr(delta, "reasoning_content", None) or (
            (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
        )
        if r:
            reasoning_parts.append(str(r))
    return "".join(content_parts).strip(), "".join(reasoning_parts).strip()


def assistant_message_text(message: Any) -> str:
    """
    Normal chat ``content``, or fallbacks used by some OpenAI-compatible providers
    (e.g. DeepSeek) that put the visible answer in ``reasoning_content`` / ``model_extra``.
    """
    raw = (getattr(message, "content", None) or "").strip()
    if raw:
        return raw
    rc = getattr(message, "reasoning_content", None) or (
        (getattr(message, "model_extra", None) or {}).get("reasoning_content")
    )
    if rc is not None and str(rc).strip():
        return str(rc).strip()
    return ""


def openai_client_from_llm_config(llm_config: Dict[str, Any]) -> OpenAI:
    return OpenAI(
        api_key=llm_config["api_key"],
        base_url=(llm_config.get("base_url") or "").strip() or None,
    )
