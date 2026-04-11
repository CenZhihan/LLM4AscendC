#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    txt: pathlib.Path
    ok: bool
    returncode: int


def _iter_txts(txt_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted([p for p in txt_dir.iterdir() if p.is_file() and p.suffix == ".txt"], key=lambda p: p.name)


def _run_one(*, repo_root: pathlib.Path, txt: pathlib.Path, mode: str, clean_policy: str) -> RunResult:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "eval_operator.py"),
        "--txt",
        str(txt),
        "--mode",
        mode,
        "--clean-policy",
        clean_policy,
    ]
    p = subprocess.run(cmd, cwd=str(repo_root))
    return RunResult(txt=txt, ok=(p.returncode == 0), returncode=p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt-dir", required=True, help="Directory containing .txt bundles to run.")
    ap.add_argument("--mode", default="full", choices=["build-only", "eval-only", "full"])
    # Keep aligned with tools/eval_operator.py choices.
    ap.add_argument("--clean-policy", default="force", choices=["force", "smart"])
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--chunk-index", type=int, default=0, help="0-based chunk index: 0 runs first N, 1 runs next N, etc.")
    ap.add_argument(
        "--scratch-dir",
        default=None,
        help="Optional temp dir to copy selected .txt files into (keeps selection stable).",
    )
    args = ap.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    txt_dir = pathlib.Path(args.txt_dir).expanduser().resolve()
    if not txt_dir.is_dir():
        raise FileNotFoundError(txt_dir)

    txts = _iter_txts(txt_dir)
    if not txts:
        print(f"[chunk] no .txt files in {txt_dir}")
        return 2

    n = args.chunk_size
    i = args.chunk_index
    start = i * n
    end = min(len(txts), start + n)
    if start >= len(txts):
        print(f"[chunk] chunk-index {i} out of range: start={start}, total={len(txts)}")
        return 2

    selected = txts[start:end]

    run_dir = txt_dir
    scratch_dir: pathlib.Path | None = None
    if args.scratch_dir:
        scratch_dir = pathlib.Path(args.scratch_dir).expanduser().resolve()
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        for p in selected:
            shutil.copy2(p, scratch_dir / p.name)
        run_dir = scratch_dir
        selected = _iter_txts(run_dir)

    ok: list[RunResult] = []
    bad: list[RunResult] = []

    print(f"[chunk] total={len(txts)}  chunk={i}  range=[{start},{end})  mode={args.mode}")
    for t in selected:
        r = _run_one(repo_root=repo_root, txt=t, mode=args.mode, clean_policy=args.clean_policy)
        (ok if r.ok else bad).append(r)

    print("\n[chunk] ===== overview =====")
    print(f"[chunk] passed: {len(ok)}")
    print(f"[chunk] failed: {len(bad)}")
    if bad:
        print("[chunk] failed list:")
        for r in bad:
            print(f"  - {r.txt.name} (exit={r.returncode})")

    # no automatic continuation: stop after this chunk
    return 0 if not bad else 3


if __name__ == "__main__":
    raise SystemExit(main())

