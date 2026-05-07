"""Write meta_task.json under artifacts_cuda_agent/<op_key>/."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_meta_task_payload(
    *,
    row_index: int | None,
    ops: str | None,
    data_source: str | None,
    code: str | None,
    parquet_path: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "cuda_agent_meta_task_v1",
        "row_index": row_index,
        "ops": ops,
        "data_source": data_source,
        "parquet_path": parquet_path,
    }
    if code is not None:
        payload["code_sha256"] = sha256_text(code)
    return payload


def write_meta_task_json(art_op_dir: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    art_op_dir.mkdir(parents=True, exist_ok=True)
    out = art_op_dir / "meta_task.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
