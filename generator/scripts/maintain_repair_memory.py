#!/usr/bin/env python3
"""Maintain repair_memory canonical: rule purge + optional LLM dedup per bucket."""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.agent.agent_config import get_llm_config_compatible
from generator.repair_memory.maintenance import (
    append_removed_archive,
    apply_canonical,
    apply_from_report,
    load_canonical_records,
    load_report,
    run_maintenance,
    write_report,
)
from generator.repair_memory.paths import get_memory_root


def main() -> int:
    p = argparse.ArgumentParser(description="Purge and deduplicate repair memory canonical JSONL.")
    p.add_argument("--model", type=str, default="", help="LLM model (same as multi-round agent)")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write canonical + archive removed (default: dry-run only)",
    )
    p.add_argument("--skip-dedup", action="store_true", help="Only run rule purge (Phase 1)")
    p.add_argument("--memory-root", type=str, default="", help="Override repair_memory root")
    p.add_argument(
        "--apply-report",
        type=str,
        default="",
        help="Apply a prior maintenance report JSON (no LLM re-run). Use with --apply to write.",
    )
    p.add_argument(
        "--apply-report-mode",
        choices=("remove_list", "kept_ids"),
        default="remove_list",
        help="remove_list=drop only report.removed ids (safe if new memories were appended); "
        "kept_ids=keep only report.kept_ids (canonical must be unchanged since report)",
    )
    args = p.parse_args()

    memory_root = Path(args.memory_root).expanduser().resolve() if args.memory_root else get_memory_root()
    canonical = memory_root / "canonical" / "repair_memories.jsonl"
    archive = memory_root / "archive" / "removed_memories.jsonl"
    reports_dir = memory_root / "maintenance_reports"

    records = load_canonical_records(canonical)
    if not records:
        print(f"[WARN] No records in {canonical}")
        return 0

    if args.apply_report:
        report_path = Path(args.apply_report).expanduser().resolve()
        report = load_report(report_path)
        kept, removed_entries, warnings = apply_from_report(
            records,
            report,
            mode=args.apply_report_mode,
        )
        print(
            f"[{'APPLY' if args.apply else 'DRY-RUN'}] from report {report_path.name} "
            f"mode={args.apply_report_mode} input={len(records)} "
            f"kept={len(kept)} removed={len(removed_entries)}"
        )
        for w in warnings:
            print(f"[WARN] {w}")
        if args.apply:
            if canonical.is_file():
                bak = canonical.with_suffix(canonical.suffix + ".bak")
                shutil.copy2(canonical, bak)
                print(f"[BACKUP] {bak}")
            apply_canonical(canonical, kept)
            append_removed_archive(archive, removed_entries)
            print(f"[WROTE] {canonical} ({len(kept)} lines)")
            if removed_entries:
                print(f"[ARCHIVE] {archive} (+{len(removed_entries)} entries)")
        else:
            print("[HINT] Add --apply to write canonical from this report.")
        return 0

    llm_config = None
    if not args.skip_dedup:
        llm_config = get_llm_config_compatible(cli_model=args.model or None)

    result = run_maintenance(
        records,
        llm_config=llm_config,
        skip_dedup=args.skip_dedup,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"report_{ts}.json"
    write_report(report_path, result, apply_mode=args.apply)

    print(
        f"[{'APPLY' if args.apply else 'DRY-RUN'}] "
        f"input={len(records)} kept={len(result.kept)} removed={len(result.removed)} "
        f"(purge={result.purge_count} dedup={result.dedup_count})"
    )
    if result.dedup_skipped_buckets:
        print(f"[INFO] dedup skipped buckets: {', '.join(result.dedup_skipped_buckets)}")
    print(f"[REPORT] {report_path}")

    if args.apply:
        if canonical.is_file():
            bak = canonical.with_suffix(canonical.suffix + ".bak")
            shutil.copy2(canonical, bak)
            print(f"[BACKUP] {bak}")
        apply_canonical(canonical, result.kept)
        append_removed_archive(archive, result.removed)
        print(f"[WROTE] {canonical} ({len(result.kept)} lines)")
        if result.removed:
            print(f"[ARCHIVE] {archive} (+{len(result.removed)} entries)")
    else:
        print(
            "[HINT] Re-run with --apply to update canonical, or use "
            "--apply-report <report.json> --apply to apply this exact report later."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
