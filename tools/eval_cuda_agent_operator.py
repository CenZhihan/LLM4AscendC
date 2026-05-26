#!/usr/bin/env python3
"""
Evaluate an AscendC operator against CUDA-Agent-Ops-6K-style PyTorch reference (dataset row code).

Mirrors tools/eval_operator.py UX: pass a *.txt bundle and staging under
artifacts_cuda_agent/.../_txt_staging/<stem>/ is recreated each run.

Examples:
  python3 tools/eval_cuda_agent_operator.py --txt output/cuda_agent_ops_6k/.../ca6k_00055.txt \\
    --dataset-path data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl --row-index 55

  python3 tools/eval_cuda_agent_operator.py --txt-dir output/cuda_agent_ops_6k/.../attempt1 \\
    --dataset-path data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl \\
    --workers 4 --npu 4 --mode full --clean-policy force
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pathlib
import shutil
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common.env import (  # noqa: E402
    ensure_parallel_build_jobs,
    init_parallel_worker_os_environ,
    load_env_config,
    resolve_ascend_custom_opp_base,
)
from tools.common.operator_txt import (  # noqa: E402
    is_mkb_operator_txt_path,
    iter_mkb_operator_txts,
    list_skipped_non_mkb_txts,
)
from tools.cuda_agent_eval.constants import (  # noqa: E402
    ENV_CUDA_AGENT_ART_ROOT,
    default_cuda_agent_art_root,
    parse_row_index_from_ca6k_op_key,
)
from tools.cuda_agent_eval.dataset_snapshot import (  # noqa: E402
    check_reference_symbols,
    load_dataset_row,
    meta_payload_from_row,
    snapshot_reference_to_op_dir,
)
from tools.cuda_agent_eval.meta import write_meta_task_json  # noqa: E402
from tools.common.runner import (  # noqa: E402
    CommandTimeoutError,
    DEFAULT_EVAL_TIMEOUT_SEC,
    EVAL_TIMEOUT_ENV,
    EXIT_CODE_EVAL_TIMEOUT,
)
from tools.eval_operator import (  # noqa: E402
    _MAX_TXT_DIR_WORKERS,
    _MAX_VISIBLE_NPU,
    _artifact_group_rel_from_txt_path,
    _execute_pipeline,
    _resolve_user_path,
    _write_txt_materialize_failure_json,
    load_operator_spec,
)
from tools.txt_operator import materialize_cuda_agent_operator_from_txt  # noqa: E402

DEFAULT_SOC = "ai_core-Ascend910B2"
DEFAULT_DATASET = ROOT / "data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl"


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


def _assert_bundle_txt(txt_path: pathlib.Path) -> None:
    if is_mkb_operator_txt_path(txt_path):
        return
    raise ValueError(
        f"refusing non-bundle txt {txt_path.name!r}; "
        "use ca6k_XXXXX.txt (not *_cot, *_repair_context, *_report)"
    )


def _run_cuda_agent_txt_bundle(
    *,
    txt_path: pathlib.Path,
    dataset_path: pathlib.Path,
    row_index: int | None,
    mode: str,
    clean_policy: str,
    art_root_override: pathlib.Path | None,
    check_reference_only: bool,
    eval_timeout_sec: int | None = None,
) -> int:
    _assert_bundle_txt(txt_path)
    if row_index is None:
        row_index = parse_row_index_from_ca6k_op_key(txt_path.stem)

    art_root = _cuda_agent_art_root_for_txt_mode(txt_path, art_root_override)
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
            mode=mode,
            exc=exc,
        )
        raise

    row = load_dataset_row(dataset_path, row_index)
    snapshot_reference_to_op_dir(
        row=row,
        op_dir=op_dir,
        dataset_path=dataset_path,
        row_index=row_index,
    )
    meta_payload = meta_payload_from_row(
        row,
        dataset_path=dataset_path,
        row_index=row_index,
    )
    ok, msg = check_reference_symbols(row["code"])
    if not ok:
        print(f"[check-reference] FAILED {txt_path.name}: {msg}")
        return 1
    print(f"[check-reference] OK {txt_path.name}")

    spec = load_operator_spec(op_dir)
    art_op_dir = art_root / spec.op_key
    write_meta_task_json(art_op_dir, meta_payload)

    if check_reference_only:
        print(f"[done] check-reference-only: {txt_path.name}")
        return 0

    try:
        _execute_pipeline(
            op_dir,
            art_root=art_root,
            mode=mode,
            clean_policy=clean_policy,
            eval_timeout_sec=eval_timeout_sec,
        )
    except CommandTimeoutError:
        return EXIT_CODE_EVAL_TIMEOUT
    print(f"[done] OK: {txt_path.name}")
    return 0


def _cuda_parallel_worker_main(
    worker_id: int,
    base_opp: str,
    task_q: Any,
    result_q: Any,
    dataset_path_str: str,
    art_root_override_str: str | None,
    mode: str,
    clean_policy: str,
    npu_count: int,
    eval_timeout_sec: int | None,
) -> None:
    init_parallel_worker_os_environ(worker_id=worker_id, base_opp=base_opp, npu_count=npu_count)
    dataset_path = pathlib.Path(dataset_path_str)
    art_override = pathlib.Path(art_root_override_str) if art_root_override_str else None
    while True:
        item = task_q.get()
        if item is None:
            break
        txt_path = pathlib.Path(item)
        print(f"[batch] [w{worker_id}] === {txt_path.name} ===")
        try:
            rc = _run_cuda_agent_txt_bundle(
                txt_path=txt_path,
                dataset_path=dataset_path,
                row_index=None,
                mode=mode,
                clean_policy=clean_policy,
                art_root_override=art_override,
                check_reference_only=False,
                eval_timeout_sec=eval_timeout_sec,
            )
            result_q.put((txt_path.name, rc == 0, None))
        except Exception as e:
            print(f"[batch] FAILED {txt_path.name}: {type(e).__name__}: {e}")
            result_q.put((txt_path.name, False, str(e)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build/install/eval fused CUDA-Agent-style operator; artifacts under "
            f"env {ENV_CUDA_AGENT_ART_ROOT} or <repo>/artifacts_cuda_agent/"
        ),
        epilog=(
            "Use --txt or --txt-dir (ca6k_XXXXX.txt stems). Skips *_cot / *_repair_context / *_report. "
            "Parallel --txt-dir uses the same OPP/NPU/build-job conventions as eval_operator."
        ),
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--op", type=pathlib.Path, help="Materialized operator directory")
    src.add_argument("--txt", type=pathlib.Path, help="Single txt bundle (stem = ca6k_XXXXX)")
    src.add_argument(
        "--txt-dir",
        dest="txt_dir",
        type=pathlib.Path,
        help="Directory of ca6k_*.txt bundles; optional --workers > 1 for parallel eval",
    )
    ap.add_argument("--mode", default="full", choices=["full", "build-only", "eval-only"])
    ap.add_argument("--clean-policy", default="force", choices=["force", "smart"])
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for --txt-dir (default 1 = sequential)",
    )
    ap.add_argument(
        "--npu",
        type=int,
        default=4,
        help="Physical NPU count 0..K-1 when --workers > 1 (default 4)",
    )
    ap.add_argument(
        "--art-root",
        type=pathlib.Path,
        default=None,
        help=f"Override artifacts root (default: env {ENV_CUDA_AGENT_ART_ROOT} or artifacts_cuda_agent/)",
    )
    ap.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=None,
        help="CUDA-Agent-Ops-6K parquet or jsonl (required for --txt / --txt-dir)",
    )
    ap.add_argument("--row-index", type=int, default=None, help="Row index for single --txt")
    ap.add_argument(
        "--check-reference-only",
        action="store_true",
        help="Validate reference only (no compile/eval)",
    )
    ap.add_argument(
        "--eval-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            f"Wall-clock timeout for NPU eval (spec.py) only; default {DEFAULT_EVAL_TIMEOUT_SEC}s "
            f"or env {EVAL_TIMEOUT_ENV}. <=0 disables."
        ),
    )
    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > _MAX_TXT_DIR_WORKERS:
        raise ValueError(f"--workers must be <= {_MAX_TXT_DIR_WORKERS}")
    if args.workers > 1 and args.txt_dir is None:
        raise ValueError("--workers > 1 is only supported with --txt-dir")
    if args.workers > 1:
        if args.npu < 1 or args.npu > _MAX_VISIBLE_NPU:
            raise ValueError(f"--npu must be between 1 and {_MAX_VISIBLE_NPU} when --workers > 1")

    if args.txt_dir is not None:
        dataset_path = (args.dataset_path or DEFAULT_DATASET).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        txt_dir = _resolve_user_path(args.txt_dir)
        if not txt_dir.is_dir():
            raise NotADirectoryError(txt_dir)
        skipped = list_skipped_non_mkb_txts(txt_dir)
        if skipped:
            print(
                f"[batch] skipping {len(skipped)} non-bundle txt: {', '.join(skipped[:8])}"
                + (f" ... (+{len(skipped) - 8} more)" if len(skipped) > 8 else "")
            )
        files = iter_mkb_operator_txts(txt_dir)
        if not files:
            raise RuntimeError(f"no ca6k bundle .txt files in {txt_dir}")
        print(f"[batch] evaluating {len(files)} cuda-agent bundle txt(s)")

        if args.workers == 1:
            rc = 0
            for txt_path in files:
                print(f"[batch] === {txt_path.name} ===")
                try:
                    one = _run_cuda_agent_txt_bundle(
                        txt_path=txt_path,
                        dataset_path=dataset_path,
                        row_index=None,
                        mode=args.mode,
                        clean_policy=args.clean_policy,
                        art_root_override=args.art_root,
                        check_reference_only=args.check_reference_only,
                        eval_timeout_sec=args.eval_timeout,
                    )
                    if one != 0:
                        rc = 1
                except Exception as e:
                    print(f"[batch] FAILED {txt_path.name}: {type(e).__name__}: {e}")
                    rc = 1
            return rc

        cfg_parallel = load_env_config()
        if not cfg_parallel.ascend_custom_opp_path:
            raise RuntimeError(
                "Parallel --txt-dir requires LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH "
                "(each worker uses <path>/_parallel_w<id>)"
            )
        base_opp = resolve_ascend_custom_opp_base(cfg_parallel.ascend_custom_opp_path)
        os.environ["LLM4ASCENDC_ASCEND_CUSTOM_OPP_BASE"] = base_opp
        os.environ["LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH"] = base_opp
        jobs = ensure_parallel_build_jobs(worker_count=args.workers)
        ncpu = os.cpu_count() or 16
        print(f"[batch] LLM4ASCENDC_BUILD_JOBS={jobs} (auto: {ncpu} cpus / {args.workers} workers)")
        ctx = multiprocessing.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        for f in files:
            task_q.put(str(f.resolve()))
        for _ in range(args.workers):
            task_q.put(None)
        art_override_str = str(args.art_root.resolve()) if args.art_root else ""
        procs: list[multiprocessing.Process] = []
        for wid in range(args.workers):
            p = ctx.Process(
                target=_cuda_parallel_worker_main,
                args=(
                    wid,
                    base_opp,
                    task_q,
                    result_q,
                    str(dataset_path),
                    art_override_str,
                    args.mode,
                    args.clean_policy,
                    args.npu,
                    args.eval_timeout,
                ),
            )
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        rc = 0
        for _ in range(len(files)):
            _name, ok, _err = result_q.get()
            if not ok:
                rc = 1
        return rc

    if args.txt is not None:
        dataset_path = (args.dataset_path or DEFAULT_DATASET).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        txt_path = _resolve_user_path(args.txt)
        if not txt_path.is_file():
            raise FileNotFoundError(txt_path)
        row_index = args.row_index
        if row_index is None:
            row_index = parse_row_index_from_ca6k_op_key(txt_path.stem)
        try:
            return _run_cuda_agent_txt_bundle(
                txt_path=txt_path,
                dataset_path=dataset_path,
                row_index=row_index,
                mode=args.mode,
                clean_policy=args.clean_policy,
                art_root_override=args.art_root,
                check_reference_only=args.check_reference_only,
                eval_timeout_sec=args.eval_timeout,
            )
        except BaseException:
            return 1

    assert args.op is not None
    op_dir = _resolve_op_dir(args.op)
    art_root = (
        args.art_root.resolve() if args.art_root is not None else default_cuda_agent_art_root()
    )
    spec = load_operator_spec(op_dir)
    ref_path = op_dir / "eval" / "reference_code.py"

    if args.dataset_path is not None:
        if args.row_index is None:
            ap.error("--row-index is required with --dataset-path for --op mode")
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
        write_meta_task_json(art_root / spec.op_key, meta_payload)
    elif args.check_reference_only or args.mode in ("full", "eval-only"):
        if not ref_path.is_file():
            print(
                "[error] Missing eval/reference_code.py. "
                "Pass --dataset-path and --row-index, or add the file manually.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.check_reference_only:
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

    if args.check_reference_only:
        print("[done] check-reference-only; skipping pipeline")
        return 0

    try:
        _execute_pipeline(
            op_dir,
            art_root=art_root,
            mode=args.mode,
            clean_policy=args.clean_policy,
            eval_timeout_sec=args.eval_timeout,
        )
    except CommandTimeoutError:
        return EXIT_CODE_EVAL_TIMEOUT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
