"""
Structured tool choice from the LLM (JSON), version 1.

Schema:
  {"tool": "<name>", "query": "<string>", "args": null | {...}}

The ``tool`` field is the canonical lowercase registry key (e.g. ``kb``, ``web``) or ``ANSWER``.
Plugins use lowercase_snake names registered via ``register_tool`` before ``parse_tool_mode`` / ``build_agent_app``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolChoiceV1:
    tool: str
    query: str = ""
    args: Optional[Dict[str, Any]] = None

    def model_dump(self) -> Dict[str, Any]:
        return {"tool": self.tool, "query": self.query, "args": self.args}


def extract_json_object(raw: str) -> Optional[str]:
    """Extract a single JSON object from model output (strip fences / prose)."""
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


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
    return ToolChoiceV1(tool=tool.strip(), query=q.strip(), args=args), None


def parse_tool_choice_json(raw: str) -> Tuple[Optional[ToolChoiceV1], Optional[str]]:
    """
    Parse and validate tool choice JSON.

    Returns:
        (ToolChoiceV1, None) on success
        (None, error_message) on failure
    """
    blob = extract_json_object(raw)
    if not blob:
        return None, "no JSON object found in model output"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    return _validate_choice(data)
