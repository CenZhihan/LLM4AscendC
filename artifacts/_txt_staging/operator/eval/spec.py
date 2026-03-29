#!/usr/bin/env python3
import os
import sys
import pathlib
import torch
import torch_npu  # noqa: F401

def _sync():
    try:
        torch_npu.npu.synchronize()
    except Exception:
        pass

def set_seed(seed: int):
    torch.manual_seed(seed)

def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "llm4ascendc_ops_matmul_with_transposed_b_custom")
    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    ctx = {"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}
    _ROOT = pathlib.Path(__file__).resolve().parents[4]
    mkb_ref_path = _ROOT / "mkb_reference" / "matmul" / "matmul_with_transposed_b.py"
    mkb_repo = _ROOT.parent / "LLM" / "MultiKernelBench-main"

    use_mkb = os.environ.get("LLM4ASCENDC_MKB_REFERENCE", "0").strip() == "1"
    execute_template = None
    if use_mkb:
        if mkb_repo.exists():
            sys.path.insert(0, str(mkb_repo))
        from utils.correctness import execute_template  # type: ignore

        ref_src = mkb_ref_path.read_text(encoding="utf-8")
        exec(ref_src, ctx, ctx)
    else:
        M, K, N = 256, 512, 384

        def get_init_inputs():
            return []

        def get_inputs():
            torch.manual_seed(0)
            A = torch.randn(M, K, dtype=torch.float16)
            B = torch.randn(N, K, dtype=torch.float16)
            return [A, B]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                return torch.matmul(A.float(), B.float().T)

        ctx.update({"Model": Model, "get_inputs": get_inputs, "get_init_inputs": get_init_inputs})

    exec('\nimport torch\nimport torch_npu\nimport custom_ops_lib\n\nclass ModelNew(torch.nn.Module):\n    def __init__(self) -> None:\n        super().__init__()\n\n    def forward(self, A, B):\n        return custom_ops_lib.matmul_with_transposed_b_custom(A, B)\n', ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")

    if use_mkb and execute_template is not None:
        device = torch.device("npu:0")
        synchronize = torch_npu.npu.synchronize
        correctness, info = execute_template(synchronize, device, ctx)
        if not correctness:
            print(info or "[FAIL] Output mismatch")
            return 1
        print("[eval] allclose=True [ref=MKB matmul_with_transposed_b]")
        return 0

    seed = 0
    trials = 3
    init_inputs = ctx["get_init_inputs"]()
    with torch.no_grad():
        set_seed(seed)
        original_model = ctx["Model"](*init_inputs).to("npu")
        _sync()
        set_seed(seed)
        custom_model = ctx["ModelNew"](*init_inputs).to("npu")
        _sync()
    with torch.no_grad():
        for _ in range(trials):
            inputs = [x.to("npu") if isinstance(x, torch.Tensor) else x for x in ctx["get_inputs"]()]
            _sync()
            ref_output = original_model(*inputs)
            _sync()
            new_output = custom_model(*inputs)
            _sync()
            if ref_output.shape != new_output.shape:
                print(f"[eval] shape mismatch: expected={ref_output.shape}, got={new_output.shape}")
                return 1
            if not torch.allclose(ref_output, new_output, atol=1e-2, rtol=1e-2):
                max_err = (ref_output - new_output).abs().max().item()
                print(f"[eval] allclose=False max_abs_err={max_err} [ref=custom small matmul]")
                return 1
    print("[eval] allclose=True [ref=custom small matmul]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
