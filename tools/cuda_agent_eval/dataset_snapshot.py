"""Load rows from CUDA-Agent-Ops-6K parquet/jsonl and snapshot reference_code.py."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from tools.cuda_agent_eval.constants import REF_CODE_FILENAME
from tools.cuda_agent_eval.meta import build_meta_task_payload, sha256_text


def _load_parquet_row(path: pathlib.Path, row_index: int) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    last_err: Exception | None = None
    try:
        from datasets import Dataset  # type: ignore

        ds = Dataset.from_parquet(str(path))
        if row_index < 0 or row_index >= len(ds):
            raise IndexError(f"row_index {row_index} out of range (len={len(ds)})")
        return dict(ds[row_index])
    except Exception as e:
        last_err = e

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        n = table.num_rows
        if row_index < 0 or row_index >= n:
            raise IndexError(f"row_index {row_index} out of range (rows={n})")
        return {name: table.column(name)[row_index].as_py() for name in table.column_names}
    except Exception as e:
        last_err = e

    raise RuntimeError(
        "Could not read parquet (install `datasets` or `pyarrow`). "
        f"Original error: {last_err}"
    ) from last_err


def _load_jsonl_row(path: pathlib.Path, row_index: int) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_index:
                line = line.strip()
                if not line:
                    raise ValueError(f"empty line at index {row_index}")
                return json.loads(line)
    raise IndexError(f"row_index {row_index} past end of {path}")


def load_dataset_row(dataset_path: pathlib.Path, row_index: int) -> dict[str, Any]:
    """Load one row (ops, data_source, code) from parquet or JSON Lines."""
    suf = dataset_path.suffix.lower()
    if suf == ".parquet":
        return _load_parquet_row(dataset_path, row_index)
    if suf == ".jsonl" or suf == ".json":
        return _load_jsonl_row(dataset_path, row_index)
    raise ValueError(f"unsupported dataset extension: {dataset_path}")


def snapshot_reference_to_op_dir(
    *,
    row: dict[str, Any],
    op_dir: pathlib.Path,
    dataset_path: pathlib.Path | None,
    row_index: int | None,
) -> pathlib.Path:
    """
    Write op_dir/eval/reference_code.py from row['code'].
    Returns path to reference_code.py.
    """
    code = row.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("row must contain non-empty string field 'code'")

    eval_dir = op_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ref_path = eval_dir / REF_CODE_FILENAME
    ref_path.write_text(code, encoding="utf-8")
    return ref_path


def _jsonish_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def meta_payload_from_row(
    row: dict[str, Any],
    *,
    dataset_path: pathlib.Path | None,
    row_index: int | None,
) -> dict[str, Any]:
    code = row.get("code") if isinstance(row.get("code"), str) else None
    return build_meta_task_payload(
        row_index=row_index,
        ops=_jsonish_str(row.get("ops")),
        data_source=_jsonish_str(row.get("data_source")),
        code=code,
        parquet_path=str(dataset_path.resolve()) if dataset_path else None,
    )


def check_reference_symbols(code: str) -> tuple[bool, str]:
    """
    Exec reference in an isolated namespace with torch available; require
    Model, get_inputs, get_init_inputs (same contract as vendor/mkb/correctness.execute_template).
    """
    import torch

    ctx: dict[str, Any] = {"torch": torch, "__name__": "__cuda_agent_ref_check__"}
    try:
        exec(compile(code, "<cuda_agent_reference_code>", "exec"), ctx, ctx)
    except Exception as e:
        return False, f"exec failed: {e!r}"
    for k in ("Model", "get_inputs", "get_init_inputs"):
        if k not in ctx:
            return False, f"missing symbol {k} after exec"
        if not callable(ctx[k]):
            return False, f"{k} is not callable"
    return True, ""
