from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .manifest import build_manifest_text, load_canonical_tail, manifest_cache_key
from .paths import get_memory_root, is_repair_memory_enabled
from .select import select_repair_memories

_MANIFEST_CACHE: Dict[str, Any] = {"key": None, "text": ""}


def _disabled_selection_meta() -> Dict[str, Any]:
    return {
        "memory_ids": [],
        "memory_ids_resolved": [],
        "memory_ids_dropped": [],
        "selection_rationale": "Repair memory disabled (LLM4ASCENDC_REPAIR_MEMORY is unset or 0).",
        "raw_model_output": "",
        "parse_ok": True,
        "parse_error": "",
    }


def memory_entries_for_report(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Structured rows for agent report (what the generation agent effectively saw per memory)."""
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        nl = (rec.get("natural_language") or "").strip()
        if len(nl) > 8000:
            nl = nl[:8000] + "...(truncated)"
        out.append(
            {
                "display_order": i,
                "memory_id": rec.get("memory_id", ""),
                "tier": rec.get("tier", ""),
                "confidence": rec.get("confidence", ""),
                "op_key": rec.get("op_key", ""),
                "category": rec.get("category", ""),
                "tool_mode": rec.get("tool_mode", ""),
                "eval_mode": rec.get("eval_mode", ""),
                "transition": rec.get("transition"),
                "failure_stage_before": rec.get("failure_stage_before", ""),
                "failure_stage_after": rec.get("failure_stage_after", ""),
                "error_anchors_before": (rec.get("error_anchors_before") or "")[:500],
                "error_anchors_after": (rec.get("error_anchors_after") or "")[:500],
                "natural_language": nl,
            }
        )
    return out


def format_injection_block(records: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for i, r in enumerate(records, 1):
        nl = (r.get("natural_language") or "").strip()
        tier = r.get("tier", "")
        tr = r.get("transition")
        tr_s = json.dumps(tr, ensure_ascii=False) if isinstance(tr, dict) else str(tr)
        parts.append(
            f"[Memory {i}] tier={tier} transition={tr_s}\n"
            f"anchors_after={(r.get('error_anchors_after') or '')[:200]}\n"
            f"{nl}"
        )
    text = "\n\n".join(parts)
    return text[:max_chars]


def _merge_selection_meta(
    sel: Dict[str, Any],
    *,
    memory_ids_resolved: List[str],
    memory_ids_dropped: List[str],
) -> Dict[str, Any]:
    meta: Dict[str, Any] = dict(sel)
    meta["memory_ids_resolved"] = list(memory_ids_resolved)
    meta["memory_ids_dropped"] = list(memory_ids_dropped)
    return meta


def build_retrieval_block_for_attempt(
    *,
    llm_config: Dict[str, Any],
    op: str,
    category: str,
    tool_mode: str,
    eval_mode: str,
    repair_error_logs_raw: str,
    attempt_id: int,
    max_n: int = 5,
    memory_root: Any = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (injection_text_for_prompt, structured_memory_rows_for_report, selection_debug_dict).

    ``selection_debug`` includes model ``memory_ids``, ``selection_rationale``, ``raw_model_output``,
    ``parse_ok`` / ``parse_error``, and after canonical lookup ``memory_ids_resolved`` /
    ``memory_ids_dropped`` (ids returned by the model but not found in the loaded tail window).
    """
    if not is_repair_memory_enabled():
        return "", [], _disabled_selection_meta()
    root = memory_root if memory_root is not None else get_memory_root()
    cache_key = manifest_cache_key(root)
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE.get("key") != cache_key:
        _MANIFEST_CACHE["key"] = cache_key
        _MANIFEST_CACHE["text"] = build_manifest_text(memory_root=root, max_records=500)
    manifest_text = str(_MANIFEST_CACHE.get("text") or "")
    query_parts = [
        f"op={op}",
        f"category={category}",
        f"tool_mode={tool_mode}",
        f"eval_mode={eval_mode}",
        f"attempt_id={attempt_id}",
    ]
    if repair_error_logs_raw.strip():
        query_parts.append("repair_context:\n" + repair_error_logs_raw.strip()[:8000])
    query = "\n".join(query_parts)

    sel = select_repair_memories(
        llm_config=llm_config,
        manifest_text=manifest_text,
        query_text=query,
        max_n=max_n,
    )
    ids: List[str] = list(sel.get("memory_ids") or [])
    if not ids:
        return "", [], _merge_selection_meta(sel, memory_ids_resolved=[], memory_ids_dropped=[])

    recs = load_canonical_tail(memory_root=root, max_records=2000)
    by_id = {str(r.get("memory_id")): r for r in recs if r.get("memory_id")}
    dropped = [i for i in ids if i not in by_id]
    chosen: List[Dict[str, Any]] = [by_id[i] for i in ids if i in by_id]
    resolved_ids = [str(r.get("memory_id")) for r in chosen if r.get("memory_id")]
    meta = _merge_selection_meta(
        sel, memory_ids_resolved=resolved_ids, memory_ids_dropped=dropped
    )
    if not chosen:
        return "", [], meta
    block = format_injection_block(chosen)
    return block, memory_entries_for_report(chosen), meta
