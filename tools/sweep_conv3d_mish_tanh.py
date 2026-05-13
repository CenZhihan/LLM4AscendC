#!/usr/bin/env python3
"""Sweep block_dim and TILE_ELEMS for conv3d_mish_tanh, collect speedup."""
from __future__ import annotations

import json, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TXT = REPO / "output/kernelbench165_txt/conv3d_mish_tanh.txt"
RESULT = REPO / "artifacts/kernelbench165_txt/conv3d_mish_tanh/result_conv3d_mish_tanh.json"

CONFIGS = [
    (32, 2048),   # baseline
    (48, 2048),
    (64, 2048),
    (32, 4096),
    (48, 4096),
]

ENV = """
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/workspace/ascend_custom_opp"
set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; set -u
set +u; source /root/miniconda3/etc/profile.d/conda.sh && conda activate multi-kernel-bench 2>/dev/null; set -u
"""


def patch(block_dim: int, tile_elems: int):
    text = TXT.read_text(encoding="utf-8")
    text = re.sub(r'uint32_t blockDim = \d+;', f'uint32_t blockDim = {block_dim};', text)
    text = re.sub(r'constexpr uint32_t TILE_ELEMS = \d+;', f'constexpr uint32_t TILE_ELEMS = {tile_elems};', text)
    TXT.write_text(text, encoding="utf-8")


def run_one(block_dim: int, tile_elems: int) -> dict | None:
    patch(block_dim, tile_elems)
    cmd = (
        f"{ENV}\n"
        f"python3 tools/eval_operator.py "
        f"--txt output/kernelbench165_txt/conv3d_mish_tanh.txt "
        f"--clean-policy force --mode full "
        f"--with-profiler --profiler-aic-metrics PipeUtilization --profiler-launch-count 1"
    )
    proc = subprocess.run(["bash", "-lc", cmd], cwd=REPO, capture_output=True, text=True, timeout=600)
    if RESULT.exists():
        data = json.loads(RESULT.read_text())
        perf = data.get("result", {}).get("conv3d_mish_tanh", {}).get("performance")
        return perf
    return None


def main():
    results = []
    for i, (bd, tl) in enumerate(CONFIGS):
        tag = f"bd{bd}_tl{tl}"
        print(f"\n[{i+1}/{len(CONFIGS)}] block_dim={bd}, TILE_ELEMS={tl} ...", flush=True)
        t0 = time.time()
        perf = run_one(bd, tl)
        elapsed = time.time() - t0
        row = (bd, tl, perf, elapsed)
        results.append(row)
        if perf:
            print(f"  speedup={perf['speedup']:.3f}x  custom={perf['custom_ms']:.3f}ms  ref={perf['ref_ms']:.3f}ms  ({elapsed:.0f}s)", flush=True)
        else:
            print(f"  FAILED ({elapsed:.0f}s)", flush=True)

    print("\n" + "=" * 75)
    print(f"{'block_dim':<12} {'TILE_ELEMS':<12} {'speedup':<10} {'custom_ms':<12} {'ref_ms':<12}")
    print("-" * 75)
    for bd, tl, perf, _ in results:
        p = perf or {}
        print(f"{bd:<12} {tl:<12} {p.get('speedup','N/A'):<10} {p.get('custom_ms','N/A'):<12} {p.get('ref_ms','N/A'):<12}")

    best = max((r for r in results if r[2]), key=lambda r: r[2].get("speedup", 0), default=None)
    if best:
        p = best[2]
        print(f"\nBest: block_dim={best[0]}, TILE_ELEMS={best[1]} → speedup={p['speedup']:.3f}x")

    # Restore best config
    if best:
        patch(best[0], best[1])
        print(f"Restored best config: block_dim={best[0]}, TILE_ELEMS={best[1]}")


if __name__ == "__main__":
    main()
