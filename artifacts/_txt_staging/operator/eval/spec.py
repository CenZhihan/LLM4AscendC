#!/usr/bin/env python3
import os
import sys
import pathlib
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
    from vendor.mkb.ref_paths import get_ref_py_path

    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "llm4ascendc_ops_conv2d_divide_leaky_relu")
    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    op_key = 'conv2d_divide_leaky_relu'
    ref_path = get_ref_py_path(op_key)
    ref_src = ref_path.read_text(encoding="utf-8")

    ctx = {"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}
    exec(ref_src, ctx, ctx)
    exec("\nimport torch\nimport torch_npu\nimport custom_ops_lib\n\n_FN = 'conv2d_divide_leaky_relu_custom'\n\nclass ModelNew(torch.nn.Module):\n    def __init__(self, *init_inputs):\n        super(ModelNew, self).__init__()\n\n    def forward(self, *inputs):\n        fn = getattr(custom_ops_lib, _FN)\n        return fn(*inputs)\n", ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")
    for k in ("Model", "get_inputs", "get_init_inputs"):
        if k not in ctx:
            raise RuntimeError("MKB reference for 'conv2d_divide_leaky_relu' must define " + k)

    device = torch.device("npu:0")
    synchronize = torch_npu.npu.synchronize
    correctness, info = execute_template(synchronize, device, ctx)
    if not correctness:
        print(info or "[FAIL] Output mismatch")
        return 1
    print("[eval] allclose=True [ref=MKB " + op_key + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
