from typing import Any, Dict, List, Optional

from .tiling_common import normalize_operator_text
from .tiling_constants import ALL_OPERATOR_CLASSES, BLACKLISTED_OPERATORS, CATEGORY_CLASS_MAP, CLASS_KEYWORDS


def normalize_operator_class(value: Any) -> str:
    normalized = normalize_operator_text(value)
    return CATEGORY_CLASS_MAP.get(normalized, normalized if normalized in ALL_OPERATOR_CLASSES else "unknown")


def classify_operator_for_tiling(
    *,
    args: Optional[Dict[str, Any]] = None,
    state_category: str = "",
    state_op_name: str = "",
    query: str = "",
) -> str:
    args = args or {}
    candidate = normalize_operator_class(args.get("op_type") or args.get("operator_class"))
    if candidate != "unknown":
        return candidate

    candidate = normalize_operator_class(state_category)
    if candidate != "unknown":
        return candidate

    for text in (normalize_operator_text(state_op_name), normalize_operator_text(query)):
        if not text:
            continue
        for operator_class, keywords in CLASS_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return operator_class
    return "unknown"


def find_blacklist_entry(op_name: str, query: str) -> Optional[Dict[str, Any]]:
    haystacks = [normalize_operator_text(op_name), normalize_operator_text(query)]
    for key, meta in BLACKLISTED_OPERATORS.items():
        if any(key in haystack for haystack in haystacks if haystack):
            return meta
    return None


def is_simple_broadcast(op_name: str, query: str) -> bool:
    text = " ".join(filter(None, [normalize_operator_text(op_name), normalize_operator_text(query)]))
    if not text:
        return False
    return "broadcast" in text or "bias" in text


def is_transpose_keyword(op_name: str, query: str) -> bool:
    text = " ".join(filter(None, [normalize_operator_text(op_name), normalize_operator_text(query)]))
    return "transpose" in text or "permute" in text


def is_supported_transpose_permutation(permutation: List[int]) -> bool:
    rank = len(permutation)
    if rank == 2:
        return permutation == [1, 0]
    if rank < 2:
        return False
    return permutation == list(range(rank - 2)) + [rank - 1, rank - 2]