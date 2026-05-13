#!/usr/bin/env python3
"""Write repair_memory_manifest.txt from canonical JSONL (one-way, for humans / debugging)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.repair_memory.manifest import build_manifest_text
from generator.repair_memory.paths import get_memory_root


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="", help="Output txt path (default: MEMORY_ROOT/manifest.txt)")
    p.add_argument("--max-records", type=int, default=500)
    args = p.parse_args()
    root = get_memory_root()
    text = build_manifest_text(memory_root=root, max_records=args.max_records)
    out = Path(args.out) if args.out else (root / "repair_memory_manifest.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")
    print(f"[WROTE] {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
