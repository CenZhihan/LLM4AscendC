"""
Structured tool choice from the LLM (JSON), version 1.

Schema:
    {
        "tool": "<name>",
        "query": "<string>",
        "args": null | {...},
        "thinking": null | {"goal": "...", "missing_info": "...", ...}
    }

The ``tool`` field is the canonical lowercase registry key (e.g. ``kb``, ``web``) or ``ANSWER``.
Plugins use lowercase_snake names registered via ``register_tool`` before ``parse_tool_mode`` / ``build_agent_app``.
The optional ``thinking`` object carries a compact, report-safe decision summary for tool routing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple


@dataclass
class ToolChoiceV1:
    tool: str
    query: str = ""
    args: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, str]] = None

    def model_dump(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"tool": self.tool, "query": self.query, "args": self.args}
        if self.thinking is not None:
            data["thinking"] = self.thinking
        return data


def extract_json_object(raw: str) -> Optional[str]:
    """Extract the first decodable JSON object from model output."""
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    for blob, _ in _iter_json_object_candidates(text):
        return blob
    return None


def _iter_json_object_candidates(text: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield decodable JSON-object candidates from left to right, ignoring trailing junk."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            data, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(data, dict):
            yield text[start : start + end], data
        start = text.find("{", start + 1)


def _validate_choice(data: Dict[str, Any]) -> Tuple[Optional[ToolChoiceV1], Optional[str]]:
    if not isinstance(data, dict):
        return None, "JSON root must be an object"
    tool = data.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None, 'field "tool" must be a non-empty string'
    q = data.get("query", "")
    if q is None:
        q = ""
    if not isinstance(q, str):
        return None, 'field "query" must be a string'
    args = data.get("args", None)
    if args is not None and not isinstance(args, dict):
        return None, 'field "args" must be an object or null'
    thinking = data.get("thinking", None)
    if thinking is not None:
        if not isinstance(thinking, dict):
            return None, 'field "thinking" must be an object or null'
        normalized_thinking: Dict[str, str] = {}
        for key, value in thinking.items():
            if not isinstance(key, str) or not key.strip():
                return None, 'field "thinking" keys must be non-empty strings'
            if value is None:
                normalized_thinking[key.strip()] = ""
                continue
            if not isinstance(value, str):
                return None, 'field "thinking" values must be strings or null'
            normalized_thinking[key.strip()] = value.strip()
        thinking = normalized_thinking
    return ToolChoiceV1(tool=tool.strip(), query=q.strip(), args=args, thinking=thinking), None


def parse_tool_choice_json(raw: str) -> Tuple[Optional[ToolChoiceV1], Optional[str]]:
    """
    Parse and validate tool choice JSON.

    Returns:
        (ToolChoiceV1, None) on success
        (None, error_message) on failure
    """
    text = (raw or "").strip()
    if not text:
        return None, "no JSON object found in model output"
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    first_validation_error: Optional[str] = None
    for _, data in _iter_json_object_candidates(text):
        choice, err = _validate_choice(data)
        if choice is not None:
            return choice, None
        if first_validation_error is None:
            first_validation_error = err

    return None, first_validation_error or "no JSON object found in model output"
