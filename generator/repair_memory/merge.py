from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


from .paths import get_memory_root
from .schema import SCHEMA_VERSION, validate_record


def merge_run_inbox(memory_root: Path | None, run_slug: str) -> int:
    """
    Append all inbox mem_*.jsonl (single-line files) for this run_slug into canonical
    repair_memories.jsonl under an exclusive flock. Moves processed files to merged/.
    Returns number of records appended.
    """
    if fcntl is None:
        return 0
    root = memory_root if memory_root is not None else get_memory_root()
    inbox = root / "inbox" / run_slug
    if not inbox.is_dir():
        return 0
    canonical_dir = root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical_dir / "repair_memories.jsonl"
    merged_dir = inbox / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(inbox.glob("mem_*.jsonl"))
    if not files:
        return 0

    appended = 0
    canonical.touch(exist_ok=True)
    with open(canonical, "a+", encoding="utf-8") as cf:
        fcntl.flock(cf.fileno(), fcntl.LOCK_EX)
        try:
            for fp in files:
                try:
                    raw = fp.read_text(encoding="utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    if not isinstance(obj, dict):
                        continue
                    if not validate_record(obj):
                        continue
                    cf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    cf.flush()
                    appended += 1
                    dest = merged_dir / fp.name
                    shutil.move(str(fp), str(dest))
                except Exception:
                    continue
        finally:
            fcntl.flock(cf.fileno(), fcntl.LOCK_UN)
    return appended
