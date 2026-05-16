from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .llm_util import (
    assistant_message_text,
    chat_completion_stream_content_reasoning,
    openai_client_from_llm_config,
)


SYSTEM = """You pick past repair memories that help the current AscendC kernel repair task.
Each manifest line is one memory (tab-separated fields: id, op, category, tool_mode, tier, root, symptom, summary).

CRITICAL (machine parsing):
- The first non-whitespace character of your reply MUST be `{` starting the JSON object below.
- No preamble text before that JSON. One line is enough.
- Shape (two keys, no markdown fences, no extra keys):
{"memory_ids": ["..."], "selection_rationale": "..."}

Rules for memory_ids:
- At most N ids from the manifest; each id MUST appear exactly as id=... on a manifest line (do not fabricate).

Rules for selection_rationale (required, non-empty string):
- If memory_ids is non-empty: briefly explain why each chosen id is relevant to the current failure (anchors/summary vs repair_context).
- If memory_ids is empty: briefly explain why none of the manifest lines are worth injecting for this repair (e.g. unrelated stages/ops, or empty manifest).

Selection policy (prefer recall over empty):
- If the manifest contains entries whose root, symptom, or summary plausibly matches the current failure
  (same error class, overlapping log phrases, same toolchain stage such as txt bundle / CPack /
  opbuild / tiling / NPU runtime), include their ids even when the originating op differs.
- Prefer entries whose root/summary names compile/API fixes when repair_context shows C++ or kernel errors.
- Prefer diverse, complementary hints when both are relevant.
- Return an empty memory_ids list only when every manifest line is clearly unrelated to the query
  repair_context and operator context, or the manifest is empty."""


def _iter_memory_selection_dicts(text: str) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Scan text for JSON objects that contain a ``memory_ids`` key.
    Returns ``(start_index, dict)`` in left-to-right order.
    """
    text = (text or "").strip()
    out: List[Tuple[int, Dict[str, Any]]] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = json.JSONDecoder().raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "memory_ids" in obj:
            out.append((j, obj))
        i = j + max(end, 1)
    return out


def parse_memory_selection_output(raw: str, *, max_n: int) -> Dict[str, Any]:
    """
    Parse model output into a structured selection dict (before canonical id resolution).

    Returns keys: memory_ids, selection_rationale, raw_model_output, parse_ok, parse_error.
    """
    raw = (raw or "").strip()
    base: Dict[str, Any] = {
        "memory_ids": [],
        "selection_rationale": "",
        "raw_model_output": raw,
        "parse_ok": False,
        "parse_error": "",
    }
    candidates = _iter_memory_selection_dicts(raw)
    if not candidates:
        base["parse_error"] = "no_json_object_in_output"
        base["selection_rationale"] = (
            "Model output did not contain a JSON object with a memory_ids field; cannot interpret selection."
        )
        return base

    first_start, first_obj = candidates[0]
    if not raw[:first_start].strip():
        obj = first_obj
    else:
        obj = candidates[-1][1]

    ids = obj.get("memory_ids")
    if not isinstance(ids, list):
        base["parse_error"] = "memory_ids_not_a_list"
        base["selection_rationale"] = "Field memory_ids was missing or not a JSON array."
        return base
    out_ids: List[str] = []
    for x in ids:
        if isinstance(x, str) and x.strip():
            out_ids.append(x.strip())
        if len(out_ids) >= max_n:
            break
    rationale = obj.get("selection_rationale")
    if isinstance(rationale, str) and rationale.strip():
        r_text = rationale.strip()
    else:
        r_text = "(model omitted selection_rationale)"
    base["memory_ids"] = out_ids
    base["selection_rationale"] = r_text
    base["parse_ok"] = True
    return base


def select_repair_memories(
    *,
    llm_config: Dict[str, Any],
    manifest_text: str,
    query_text: str,
    max_n: int = 5,
) -> Dict[str, Any]:
    """
    Ask the selection LLM for memory ids plus rationale; always returns a dict suitable for reports.

    Uses **streaming** completions (same as the main generation agent) so providers that only
    fill ``delta.content`` in stream mode deliver the final JSON. Falls back to non-streaming
    + ``assistant_message_text`` if streaming yields nothing.

    Keys: memory_ids, selection_rationale, raw_model_output, parse_ok, parse_error.
    """
    if not (manifest_text or "").strip():
        return {
            "memory_ids": [],
            "selection_rationale": (
                "Manifest text was empty (no candidate memories in canonical tail after schema filter)."
            ),
            "raw_model_output": "",
            "parse_ok": True,
            "parse_error": "",
        }
    client = openai_client_from_llm_config(llm_config)
    model = llm_config.get("model") or ""
    user = (
        f"Select up to {max_n} memory ids and explain your choice.\n\n"
        f"=== Manifest ===\n{manifest_text}\n\n=== Current query ===\n{query_text[:12000]}"
    )
    messages = [
        {"role": "system", "content": SYSTEM.replace("N", str(max_n))},
        {"role": "user", "content": user},
    ]
    try:
        content, reasoning = chat_completion_stream_content_reasoning(
            client,
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        raw = (content or "").strip() or (reasoning or "").strip()
        if not raw:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0.1,
                max_tokens=4096,
            )
            raw = assistant_message_text(resp.choices[0].message)
    except Exception as exc:  # noqa: BLE001 — surface to report instead of silent []
        return {
            "memory_ids": [],
            "selection_rationale": f"Selection LLM request failed: {exc!s}",
            "raw_model_output": "",
            "parse_ok": False,
            "parse_error": str(exc),
        }
    parsed = parse_memory_selection_output(raw, max_n=max_n)
    return parsed


def select_memory_ids(
    *,
    llm_config: Dict[str, Any],
    manifest_text: str,
    query_text: str,
    max_n: int = 5,
) -> List[str]:
    """Backward-compatible wrapper: returns only the id list."""
    return list(
        select_repair_memories(
            llm_config=llm_config,
            manifest_text=manifest_text,
            query_text=query_text,
            max_n=max_n,
        ).get("memory_ids")
        or []
    )
