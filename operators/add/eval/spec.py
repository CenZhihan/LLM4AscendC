#!/usr/bin/env python3
import os

import torch
import torch_npu  # noqa: F401


def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "llm4ascendc_ops_add")
    mod = __import__(module_name)

    torch.manual_seed(0)
    a = torch.randn(4096, device="npu", dtype=torch.float32)
    b = torch.randn(4096, device="npu", dtype=torch.float32)

    out_custom = mod.add_custom(a, b)
    out_ref = a + b

    ok = torch.allclose(out_custom, out_ref, rtol=1e-4, atol=1e-5)
    max_err = (out_custom - out_ref).abs().max().item()
    print(f"[eval:{module_name}] allclose={ok}, max_abs_err={max_err:.8f}")
    if ok:
        print("test pass!")
        return 0
    print("test failed!")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

