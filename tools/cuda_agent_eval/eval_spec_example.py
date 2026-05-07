#!/usr/bin/env python3
"""
Example CUDA-Agent fused-task eval/spec.py pattern.

Copy this file to your materialized operator's ``eval/spec.py`` after you provide:

- ``eval/reference_code.py`` — snapshot of dataset ``code`` (defines Model, get_inputs, get_init_inputs).
- ``eval/model_new.py`` — defines ``ModelNew`` using the pybind module as ``custom_ops_lib``.

This mirrors vendor/mkb single-op behaviour but loads reference from disk instead of get_ref_py_path.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch_npu  # noqa: F401


def _find_repo_root() -> pathlib.Path:
    p = pathlib.Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vendor" / "mkb" / "correctness.py").is_file():
            return parent
    raise RuntimeError(
        "Cannot find LLM4AscendC repo root (expected vendor/mkb/correctness.py on a parent path)."
    )


def main() -> int:
    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vendor.mkb.correctness import execute_template

    op_dir = pathlib.Path(os.environ.get("LLM4ASCENDC_OP_DIR", ".")).resolve()
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "custom_ops_lib")

    ref_path = op_dir / "eval" / "reference_code.py"
    if not ref_path.is_file():
        raise RuntimeError(f"missing {ref_path}; snapshot dataset row or copy code here")

    model_new_path = op_dir / "eval" / "model_new.py"
    if not model_new_path.is_file():
        raise RuntimeError(
            f"missing {model_new_path}; provide ModelNew that uses custom_ops_lib"
        )

    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    ref_src = ref_path.read_text(encoding="utf-8")
    mn_src = model_new_path.read_text(encoding="utf-8")

    ctx: dict = {"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}
    exec(compile(ref_src, str(ref_path), "exec"), ctx, ctx)
    exec(compile(mn_src, str(model_new_path), "exec"), ctx, ctx)

    if "ModelNew" not in ctx:
        raise RuntimeError("model_new.py must define ModelNew")
    for k in ("Model", "get_inputs", "get_init_inputs"):
        if k not in ctx:
            raise RuntimeError(f"reference_code.py must define {k}")

    device = torch.device("npu:0")
    synchronize = torch_npu.npu.synchronize
    correctness, info = execute_template(synchronize, device, ctx)
    if not correctness:
        print(info or "[FAIL] Output mismatch")
        return 1
    print("[eval] allclose=True [ref=CUDA-Agent-Ops-6K reference_code.py]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
