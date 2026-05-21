from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .paths import get_memory_root


def write_memory_inbox_line(record: dict, *, run_slug: str) -> Path:
    """
    Atomically write one JSON line into a new unique file under inbox/<run_slug>/.
    """
    root = get_memory_root()
    inbox = root / "inbox" / run_slug
    inbox.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    path = inbox / f"mem_{uid}.jsonl"
    line = json.dumps(record, ensure_ascii=False) + "\n"
    tmp = inbox / f".tmp_{os.getpid()}_{uid}.jsonl"
    tmp.write_text(line, encoding="utf-8")
    os.replace(str(tmp), str(path))
    return path
