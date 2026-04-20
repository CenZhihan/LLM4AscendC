"""Helpers for parsing structured tool arguments and API-oriented queries."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple


_META_API_WORDS = {
    "a",
    "an",
    "alternative",
    "alternatives",
    "api",
    "apis",
    "check",
    "constraint",
    "constraints",
    "detail",
    "details",
    "docs",
    "documentation",
    "exact",
    "exist",
    "exists",
    "for",
    "header",
    "headers",
    "lookup",
    "of",
    "query",
    "s",
    "signature",
    "signatures",
    "symbol",
    "symbols",
    "the",
    "usage",
    "usages",
}

_NPU_QUERY_KEYWORDS = (
    ("memory", ("memory", "mem", "显存", "内存")),
    ("temp", ("temp", "temperature", "温度")),
    ("power", ("power", "功耗")),
    ("usages", ("usage", "usages", "utilization", "利用率", "负载")),
    ("list", ("list", "列表", "设备列表")),
    ("info", ("info", "status", "状态", "概览", "overview")),
)


def get_tool_args(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return parsed tool args from the last tool choice JSON."""
    tool_choice = state.get("tool_choice_json") or {}
    args = tool_choice.get("args")
    return args if isinstance(args, dict) else {}


def _clean_symbol(token: str) -> str:
    return (token or "").strip().strip("`\"'.,:;()[]{}<>")


def _is_meta_word(token: str) -> bool:
    return token.strip().lower() in _META_API_WORDS


def _canonical_name(token: str) -> str:
    value = _clean_symbol(token)
    for prefix in ("AscendC::", "ascendc::"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def extract_api_name(
    query: str,
    *,
    args: Optional[Dict[str, Any]] = None,
    known_names: Optional[Iterable[str]] = None,
) -> str:
    """Extract the most likely concrete API symbol from tool input."""
    if args:
        for key in ("api_name", "symbol", "name"):
            value = args.get(key)
            if isinstance(value, str):
                cleaned = _clean_symbol(value)
                if cleaned and not _is_meta_word(cleaned):
                    return cleaned

    text = (query or "").strip()
    if not text:
        return "unknown"

    known = set()
    if known_names:
        for name in known_names:
            cleaned = _clean_symbol(name)
            if cleaned:
                known.add(cleaned)
                known.add(_canonical_name(cleaned))

    candidates = []
    explicit_patterns = [
        r"(AscendC::[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)",
        r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)",
        r"(?:api(?:\s+(?:name|symbol))?\s*[:=]\s*|signature\s+(?:of|for)\s+|constraints?\s+(?:for|of)\s+|alternative\s+(?:for)\s+|check\s*(?:if\s*)?(?:api\s*)?[:=]?\s*)([A-Za-z_][\w:]*)",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidates.append(match.group(1))
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_:]{2,}\b", text):
        candidates.append(match.group(0))

    scored = []
    seen = set()
    for raw in candidates:
        candidate = _clean_symbol(raw)
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)

        bare = _canonical_name(candidate)
        if _is_meta_word(candidate) or _is_meta_word(bare):
            continue

        score = 0
        if candidate in known or bare in known:
            score += 100
        if "::" in candidate:
            score += 50
        if re.search(r"[A-Z]", candidate):
            score += 20
        if re.search(r"(?:Copy|Pad|Mul|Add|Sub|Div|Reduce|Pool|Max|Min|Cast|Type|Tiling|Tensor)$", bare):
            score += 10
        if candidate.lower().startswith("ascendc::"):
            score += 5
        scored.append((score, candidate))

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1]

    for part in (_clean_symbol(part) for part in text.split()):
        if part and not _is_meta_word(part):
            return part
    return "unknown"


def extract_npu_query_params(
    query: str,
    *,
    args: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[int]]:
    """Extract NPU query type and optional device id from tool input."""
    query_type = "info"
    device_id: Optional[int] = None

    if args:
        raw_query_type = args.get("query_type")
        if isinstance(raw_query_type, str):
            raw_query_type = raw_query_type.strip().lower()
            if raw_query_type in {name for name, _ in _NPU_QUERY_KEYWORDS}:
                query_type = raw_query_type
        raw_device_id = args.get("device_id")
        if isinstance(raw_device_id, int) and raw_device_id >= 0:
            device_id = raw_device_id

    text = (query or "").strip().lower()
    for candidate, keywords in _NPU_QUERY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            query_type = candidate
            break

    if device_id is None:
        match = re.search(r"(?:device|card|npu)\s*#?\s*(\d+)", text)
        if not match:
            match = re.search(r"-i\s*(\d+)", text)
        if match:
            device_id = int(match.group(1))

    return query_type, device_id
