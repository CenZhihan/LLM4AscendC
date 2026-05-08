"""Resolve CUDA-Agent-Ops-6K jsonl row indices for batch generation (CLI filters)."""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, Optional


def parse_ops_len(ops_field: str) -> int:
    """Return count of API names in dataset ``ops`` field (JSON array string)."""
    return len(json.loads(ops_field))


def _iter_jsonl_rows(path: pathlib.Path) -> Iterable[tuple[int, dict]]:
    """Yield (line_index, parsed_row) for non-empty lines."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def resolve_row_indices(
    dataset_path: pathlib.Path,
    *,
    indices: Optional[list[int]] = None,
    range_pair: Optional[tuple[int, int]] = None,
    use_all: bool = False,
    op_counts: Optional[list[int]] = None,
) -> list[int]:
    """
    Return sorted unique row indices (0-based) into jsonl.

    Exactly one selection mode must be set:
    - ``indices``: explicit row numbers
    - ``range_pair``: closed interval [start, end]
    - ``use_all``: every row in the file

    If ``op_counts`` is non-empty, keep only rows whose ``len(json.loads(ops))`` is in that set.
    """
    modes = sum(
        1
        for x in (
            indices is not None,
            range_pair is not None,
            use_all,
        )
        if x
    )
    if modes != 1:
        raise ValueError(
            "Specify exactly one of: --indices, --range START END, or --all"
        )

    if indices is not None:
        if not indices:
            raise ValueError("--indices requires at least one row index")
        for x in indices:
            if x < 0:
                raise ValueError(f"row index must be >= 0, got {x}")
        index_set = set(indices)
    elif range_pair is not None:
        start, end = range_pair
        if start < 0 or end < 0:
            raise ValueError("range bounds must be >= 0")
        if start > end:
            raise ValueError(f"range start ({start}) must be <= end ({end})")
        index_set = None
    else:
        index_set = None

    allowed = set(op_counts) if op_counts else None

    out: list[int] = []
    max_line_idx = -1

    for i, row in _iter_jsonl_rows(dataset_path):
        max_line_idx = max(max_line_idx, i)

        if index_set is not None:
            if i not in index_set:
                continue
        elif range_pair is not None:
            start, end = range_pair
            if i < start or i > end:
                continue

        if allowed is not None:
            n = parse_ops_len(row["ops"])
            if n not in allowed:
                continue

        out.append(i)

    if indices is not None:
        too_large = [x for x in indices if x > max_line_idx]
        if too_large:
            raise ValueError(
                f"indices past end of dataset (max row index = {max_line_idx}): "
                f"{too_large[:20]!r}{'...' if len(too_large) > 20 else ''}"
            )

    return sorted(set(out))
