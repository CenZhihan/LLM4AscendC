"""CUDA-Agent multi-op harness: paths and naming conventions."""

from __future__ import annotations

import os
import pathlib

# Override default artifact root for fused CUDA-Agent-style evaluation (parallel to artifacts/).
ENV_CUDA_AGENT_ART_ROOT = "LLM4ASCENDC_CUDA_AGENT_ART_ROOT"

DEFAULT_ART_SUBDIR = "artifacts_cuda_agent"


def resolve_repo_root() -> pathlib.Path:
    """Return LLM4AscendC root (directory containing vendor/mkb/)."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vendor" / "mkb" / "correctness.py").is_file():
            return parent
    raise RuntimeError(
        "Cannot find repo root (expected vendor/mkb/correctness.py on a parent path)."
    )


def default_cuda_agent_art_root(repo_root: pathlib.Path | None = None) -> pathlib.Path:
    root = repo_root or resolve_repo_root()
    env = os.environ.get(ENV_CUDA_AGENT_ART_ROOT, "").strip()
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return (root / DEFAULT_ART_SUBDIR).resolve()


def suggested_op_key_ca6k(row_index: int) -> str:
    """Stable id when materializing from CUDA-Agent-Ops-6K row index (0-based)."""
    if row_index < 0:
        raise ValueError("row_index must be >= 0")
    return f"ca6k_{row_index:05d}"


REF_CODE_FILENAME = "reference_code.py"
MODEL_NEW_FILENAME = "model_new.py"
META_TASK_FILENAME = "meta_task.json"
