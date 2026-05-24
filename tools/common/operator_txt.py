#!/usr/bin/env python3
"""Shared rules for MKB / agent *.txt bundle paths (eval + batch runners)."""
from __future__ import annotations

import pathlib


def is_mkb_operator_txt_path(p: pathlib.Path) -> bool:
    """
    True for operator MKB bundle *.txt.

    Skips:
    - CoT sidecars (*_cot.txt)
    - repair-memory context dumps (*_repair_context.txt)
    - report sidecars (*_report.txt) when present next to bundles
    """
    if not (p.is_file() and p.suffix == ".txt"):
        return False
    stem = p.stem
    if stem.endswith("_cot"):
        return False
    if stem.endswith("_repair_context") or "_repair_context" in stem:
        return False
    if stem.endswith("_report"):
        return False
    return True


def iter_mkb_operator_txts(txt_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in txt_dir.glob("*.txt") if is_mkb_operator_txt_path(p))


def list_skipped_non_mkb_txts(txt_dir: pathlib.Path) -> list[str]:
    return sorted(
        p.name
        for p in txt_dir.glob("*.txt")
        if p.is_file() and not is_mkb_operator_txt_path(p)
    )
