#!/usr/bin/env python3
"""PyTorch NPU profiler for conv3d_mish_tanh — captures per-kernel timing breakdown.

Unlike msprof op (single-op profiling), this captures the full execution graph
including framework overhead, kernel launch overhead, and all intermediate kernels.
Output: kernel_details.csv + tensorboard traces under artifacts/.../profiler_current/
"""
import pathlib, sys, torch, torch_npu
from torch_npu.profiler import (
    AiCMetrics,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    schedule,
    tensorboard_trace_handler,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/kernelbench165_txt/conv3d_mish_tanh/profiler_current"

SHAPE = (16, 64, 30, 62, 62)  # typical Conv3D output, ~118M elems


def profile_one(label: str, fn, x: torch.Tensor):
    print(f"\n=== Profiling {label} ===", flush=True)
    prof_dir = OUT / label
    prof_dir.mkdir(parents=True, exist_ok=True)

    exp = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level2,
        aic_metrics=AiCMetrics.PipeUtilization,
        data_simplification=False,
    )

    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(prof_dir), analyse_flag=True),
            record_shapes=True,
            experimental_config=exp,
        ) as prof:
            for _ in range(7):
                fn(x)
                torch_npu.npu.synchronize()
                prof.step()

    # Find kernel_details.csv
    for csv in sorted(prof_dir.rglob("kernel_details.csv")):
        print(f"  -> {csv}")
    print(f"  profiler output: {prof_dir}")
    return prof_dir


def load_custom_op():
    custom_ops = __import__("llm4ascendc_ops_conv3d_mish_tanh")
    return getattr(custom_ops, "conv3d_mish_tanh_custom")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device("npu:0")
    x = torch.randn(*SHAPE, dtype=torch.float32, device=device)
    print(f"Input: {list(x.shape)}  dtype={x.dtype}  device={x.device}  nelem={x.numel():,}")

    # 1) PyTorch reference (softplus + tanh + tanh as separate ops)
    def ref_fn(t: torch.Tensor) -> torch.Tensor:
        sp = torch.nn.functional.softplus(t)
        m = t * torch.tanh(sp)
        return torch.tanh(m)

    # 2) Our fused custom op
    fn = load_custom_op()

    # Warmup
    for _ in range(3):
        ref_fn(x)
        torch_npu.npu.synchronize()
        fn(x)
        torch_npu.npu.synchronize()

    # Verify correctness
    y_custom = fn(x)
    y_ref = ref_fn(x)
    if torch.allclose(y_custom, y_ref, atol=1e-4, rtol=1e-4):
        print("[OK] allclose=True")
    else:
        md = (y_custom - y_ref).abs().max().item()
        print(f"[FAIL] max_diff={md}")
        return 1

    # Profile both
    profile_one("ref", ref_fn, x)
    profile_one("custom", fn, x)

    print("\nDone. Process with:")
    print(f"  python3 tools/visualize_gelu_profile.py --ref-dir {OUT}/ref --custom-dir {OUT}/custom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
