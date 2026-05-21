#!/usr/bin/env python3
"""Apply hand-authored NL patches to canonical repair_memories.jsonl."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.repair_memory.maintenance import load_canonical_records
from generator.repair_memory.nl_patches import NL_PATCHES
from generator.repair_memory.paths import get_memory_root
from generator.repair_memory.review_llm import _has_scenario_scope, _has_template_parts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Write canonical (default: dry-run)")
    p.add_argument("--memory-root", type=str, default="")
    args = p.parse_args()

    root = Path(args.memory_root).expanduser().resolve() if args.memory_root else get_memory_root()
    canonical = root / "canonical" / "repair_memories.jsonl"
    records = load_canonical_records(canonical)
    if not records:
        print(f"[WARN] empty or missing {canonical}")
        return 1

    changed = []
    missing = []
    for rec in records:
        mid = rec.get("memory_id", "")
        if mid not in NL_PATCHES:
            continue
        new_nl = NL_PATCHES[mid]
        old_nl = rec.get("natural_language", "")
        if old_nl.strip() == new_nl.strip():
            continue
        if not _has_template_parts(new_nl):
            print(f"[ERROR] patch for {mid} fails template check")
            return 1
        if not _has_scenario_scope(new_nl):
            print(f"[ERROR] patch for {mid} fails scenario scope check")
            return 1
        rec["natural_language"] = new_nl
        changed.append((mid, rec.get("op_key"), len(old_nl), len(new_nl)))

    for mid in NL_PATCHES:
        if not any(r.get("memory_id") == mid for r in records):
            missing.append(mid)

    print(f"[{'APPLY' if args.apply else 'DRY-RUN'}] patches defined={len(NL_PATCHES)} applied={len(changed)} missing_ids={len(missing)}")
    for mid, op, olen, nlen in changed:
        print(f"  {mid[:8]} op={op} nl_len {olen} -> {nlen}")

    if missing:
        print("[WARN] patch ids not in canonical:", ", ".join(m[:8] for m in missing))

    if not changed:
        print("[OK] nothing to change")
        return 0

    if args.apply:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = canonical.with_suffix(canonical.suffix + f".pre_nl_patch_{ts}.bak")
        shutil.copy2(canonical, bak)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        canonical.write_text(text, encoding="utf-8")
        print(f"[BACKUP] {bak}")
        print(f"[WROTE] {canonical}")
    else:
        print("[HINT] Re-run with --apply to write canonical.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
