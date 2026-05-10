#!/usr/bin/env python3
"""
Evaluate an AscendC operator against CUDA-Agent-Ops-6K-style PyTorch reference (dataset row code).

Mirrors tools/eval_operator.py UX: pass a *.txt bundle and staging under
artifacts_cuda_agent/.../_txt_staging/<stem>/ is recreated each run.

Does NOT modify tools/eval_operator.py.

Examples:
  python3 tools/eval_cuda_agent_operator.py --txt output/cuda_agent_ops_6k/ca6k_rh_fused.txt \\
    --dataset-path data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl --row-index 55
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cuda_agent_eval.constants import (  # noqa: E402
    ENV_CUDA_AGENT_ART_ROOT,
    default_cuda_agent_art_root,
)
from tools.cuda_agent_eval.dataset_snapshot import (  # noqa: E402
    check_reference_symbols,
    load_dataset_row,
    meta_payload_from_row,
    snapshot_reference_to_op_dir,
)
from tools.cuda_agent_eval.meta import write_meta_task_json  # noqa: E402
from tools.eval_operator import (  # noqa: E402
    _artifact_group_rel_from_txt_path,
    _execute_pipeline,
    _resolve_user_path,
    _write_txt_materialize_failure_json,
    load_operator_spec,
)
from tools.txt_operator import materialize_cuda_agent_operator_from_txt  # noqa: E402

DEFAULT_SOC = "ai_core-Ascend910B2"


def _resolve_op_dir(p: pathlib.Path) -> pathlib.Path:
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_dir():
        raise NotADirectoryError(p)
    return p


def _cuda_agent_art_root_for_txt_mode(
    txt_path: pathlib.Path, art_root_override: pathlib.Path | None
) -> pathlib.Path:
    if art_root_override is not None:
        return art_root_override.resolve()
    group_rel = _artifact_group_rel_from_txt_path(txt_path)
    base = default_cuda_agent_art_root()
    return (base / group_rel) if group_rel else base


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build/install/eval fused CUDA-Agent-style operator; artifacts under "
            f"env {ENV_CUDA_AGENT_ART_ROOT} or <repo>/artifacts_cuda_agent/"
        ),
        epilog=(
            "Use --txt to materialize from bundle each run (_txt_staging/<stem>/ is wiped first). "
            "For --mode full or eval-only, supply --dataset-path + --row-index unless "
            "eval/reference_code.py already exists under the operator directory."
        ),
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--op",
        type=pathlib.Path,
        help="Materialized operator directory",
    )
    src.add_argument(
        "--txt",
        type=pathlib.Path,
        help="Txt bundle path (stem = op_key); writes _txt_staging/<stem>/operator/",
    )
    ap.add_argument("--mode", default="full", choices=["full", "build-only", "eval-only"])
    ap.add_argument("--clean-policy", default="force", choices=["force", "smart"])
    ap.add_argument(
        "--art-root",
        type=pathlib.Path,
        default=None,
        help=f"Override artifacts root (default: env {ENV_CUDA_AGENT_ART_ROOT} or <repo>/artifacts_cuda_agent/)",
    )
    ap.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=None,
        help="CUDA-Agent-Ops-6K parquet or jsonl",
    )
    ap.add_argument("--row-index", type=int, default=None, help="Row index for --dataset-path")
    ap.add_argument(
        "--check-reference-only",
        action="store_true",
        help="Validate reference only (no compile/eval)",
    )
    args = ap.parse_args()

    if (args.dataset_path is None) != (args.row_index is None):
        ap.error("--dataset-path and --row-index must be passed together")

    art_root: pathlib.Path
    op_dir: pathlib.Path

    if args.txt is not None:
        txt_path = _resolve_user_path(pathlib.Path(args.txt))
        if not txt_path.is_file():
            raise FileNotFoundError(txt_path)
        if txt_path.stem.endswith("_cot"):
            sugg = txt_path.stem.removesuffix("_cot") + ".txt"
            raise ValueError(
                f"refusing CoT sidecar {txt_path.name!r}; use bundle e.g. {sugg!r}"
            )

        art_root = _cuda_agent_art_root_for_txt_mode(txt_path, args.art_root)
        staging_root = art_root / "_txt_staging" / txt_path.stem
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        try:
            op_dir = materialize_cuda_agent_operator_from_txt(
                out_dir=staging_root / "operator",
                txt_path=txt_path,
                soc=DEFAULT_SOC,
            )
        except BaseException as exc:
            _write_txt_materialize_failure_json(
                art_root=art_root,
                op_key=txt_path.stem,
                mode=args.mode,
                exc=exc,
            )
            raise SystemExit(1)
    else:
        assert args.op is not None
        op_dir = _resolve_op_dir(args.op)
        art_root = (
            args.art_root.resolve()
            if args.art_root is not None
            else default_cuda_agent_art_root()
        )

    spec = load_operator_spec(op_dir)
    ref_path = op_dir / "eval" / "reference_code.py"

    meta_payload = None
    if args.dataset_path is not None:
        assert args.row_index is not None
        row = load_dataset_row(args.dataset_path, args.row_index)
        snapshot_reference_to_op_dir(
            row=row,
            op_dir=op_dir,
            dataset_path=args.dataset_path,
            row_index=args.row_index,
        )
        meta_payload = meta_payload_from_row(
            row,
            dataset_path=args.dataset_path,
            row_index=args.row_index,
        )
        ok, msg = check_reference_symbols(row["code"])
        if not ok:
            print(f"[check-reference] FAILED: {msg}")
            return 1
        print("[check-reference] OK: Model, get_inputs, get_init_inputs present")
    elif args.check_reference_only or args.mode in ("full", "eval-only"):
        if not ref_path.is_file():
            print(
                "[error] Missing eval/reference_code.py. "
                "Pass --dataset-path and --row-index, or add the file manually.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.dataset_path is None and args.check_reference_only:
            ok, msg = check_reference_symbols(ref_path.read_text(encoding="utf-8"))
            if not ok:
                print(f"[check-reference] FAILED: {msg}")
                return 1
            print("[check-reference] OK: Model, get_inputs, get_init_inputs present")

    if args.mode in ("full", "eval-only") and not args.check_reference_only:
        if not ref_path.is_file():
            print(
                "[error] eval requires eval/reference_code.py. "
                "Use --dataset-path + --row-index or add reference_code.py.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    art_op_dir = art_root / spec.op_key
    if meta_payload is not None:
        write_meta_task_json(art_op_dir, meta_payload)

    if args.check_reference_only:
        print("[done] check-reference-only; skipping pipeline")
        return 0

    _execute_pipeline(
        op_dir,
        art_root=art_root,
        mode=args.mode,
        clean_policy=args.clean_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
