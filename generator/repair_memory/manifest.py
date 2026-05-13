from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .anchors import extract_anchor
from .paths import get_memory_root
from .schema import SCHEMA_VERSION


def load_canonical_tail(*, memory_root: Path | None = None, max_records: int = 500) -> List[Dict[str, Any]]:
    root = memory_root if memory_root is not None else get_memory_root()
    path = root / "canonical" / "repair_memories.jsonl"
    if not path.is_file():
        return []
    lines: List[str] = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = min(size, 512 * 1024)
        f.seek(max(0, size - block))
        if f.tell() > 0:
            f.readline()  # drop partial
        tail = f.read().decode("utf-8", errors="replace")
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for ln in lines[-max_records:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def build_manifest_lines(
    records: List[Dict[str, Any]],
    *,
    tool_mode_filter: str = "",
    eval_mode_filter: str = "",
) -> List[str]:
    """One line per record for the selection model."""
    lines: List[str] = []
    for r in records:
        if (r.get("schema_version") or "") != SCHEMA_VERSION:
            continue
        if tool_mode_filter and (r.get("tool_mode") or "") != tool_mode_filter:
            continue
        if eval_mode_filter and (r.get("eval_mode") or "") != eval_mode_filter:
            continue
        mid = r.get("memory_id", "")
        op = r.get("op_key", "")
        cat = r.get("category", "")
        tm = r.get("tool_mode", "")
        tier = r.get("tier", "")
        nl = (r.get("natural_language") or "").replace("\n", " ").strip()
        if len(nl) > 160:
            nl = nl[:157] + "..."
        anchor = extract_anchor(r.get("error_anchors_after") or r.get("correctness_info", "") or nl)
        lines.append(f"id={mid}\top={op}\tcategory={cat}\ttool_mode={tm}\ttier={tier}\tanchor={anchor[:120]}\tsummary={nl}")
    return lines


def build_manifest_text(**kwargs: Any) -> str:
    mem_root = kwargs.get("memory_root")
    max_records = int(kwargs.get("max_records", 500))
    tool_mode_filter = str(kwargs.get("tool_mode_filter", "") or "")
    eval_mode_filter = str(kwargs.get("eval_mode_filter", "") or "")
    recs = load_canonical_tail(memory_root=mem_root, max_records=max_records)
    lines = build_manifest_lines(
        recs, tool_mode_filter=tool_mode_filter, eval_mode_filter=eval_mode_filter
    )
    return "\n".join(lines)


def manifest_cache_key(memory_root: Path) -> Tuple[float, int]:
    p = memory_root / "canonical" / "repair_memories.jsonl"
    if not p.is_file():
        return (0.0, 0)
    st = p.stat()
    return (st.st_mtime, st.st_size)
