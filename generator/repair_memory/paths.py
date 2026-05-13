from __future__ import annotations

import os
from pathlib import Path

from generator.repo_root import REPO_ROOT


def is_repair_memory_enabled() -> bool:
    v = (os.environ.get("LLM4ASCENDC_REPAIR_MEMORY") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def get_memory_root() -> Path:
    override = (os.environ.get("LLM4ASCENDC_REPAIR_MEMORY_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (REPO_ROOT / "repair_memory").resolve()


def run_slug_from_run_dir(run_dir: Path) -> str:
    """Stable slug for inbox subdir (no raw slashes in single segment)."""
    try:
        rel = run_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        rel = run_dir.resolve()
    parts = [p for p in rel.parts if p not in (".", "/") and p]
    slug = "__".join(parts) if parts else "unknown_run"
    if len(slug) > 200:
        slug = slug[:200]
    for ch in ':*?"<>|':
        slug = slug.replace(ch, "_")
    return slug or "unknown_run"
