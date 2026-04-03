#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

# Upper bound for --workers (txt-dir parallel mode) to avoid accidental huge values.
_MAX_TXT_DIR_WORKERS = 32

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common.env import EnvConfig, build_subprocess_env, load_env_config, shell_prefix  # noqa: E402
from tools.common.fingerprint import compute_fingerprint, read_fingerprint, write_fingerprint  # noqa: E402
from tools.common.runner import now_tag, run_cmd  # noqa: E402
from tools.txt_operator import materialize_operator_from_txt  # noqa: E402


OPS_ROOT = ROOT / "operators"
ART_ROOT = ROOT / "artifacts"
OUTPUT_ROOT = ROOT / "output"
TOOLS_ROOT = ROOT / "tools"


@dataclass(frozen=True)
class OperatorSpec:
    op_key: str
    op_name: str
    op_snake: str
    soc: str
    project_json: list[dict]
    pybind: dict


def _read_tail(path: pathlib.Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n".join(tail)


def _extract_core_error(text: str) -> str:
    """
    Heuristic: pick the last traceback / error-looking lines.
    Keep it short and high-signal for humans.
    """
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    key_markers = (
        "Traceback (most recent call last):",
        "RuntimeError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "CalledProcessError:",
        "CMake Error",
        "error:",
        "ERROR",
        "ERR",
        "No such file or directory",
        "not found",
        "failed",
    )
    idxs: list[int] = [i for i, ln in enumerate(lines) if any(m in ln for m in key_markers)]
    if not idxs:
        return "\n".join(lines[-20:])
    start = max(0, idxs[-1] - 6)
    end = min(len(lines), idxs[-1] + 8)
    return "\n".join(lines[start:end])


def write_result_json(
    *,
    art_dir: pathlib.Path,
    op_key: str,
    compiled: bool | None,
    correctness: bool | None,
    correctness_info: str | None,
    logs: dict[str, str],
    fingerprint: str | None,
    mode: str,
) -> pathlib.Path:
    """
    Overwrite a single JSON file each run.
    Format is compatible with the referenced MKB-style top-level map,
    but we store only the current operator.
    """
    out = {
        op_key: {
            "compiled": compiled,
            "correctness": correctness,
            "performance": None,
            "correctness_info": correctness_info,
        }
    }
    meta: dict[str, Any] = {
        "mode": mode,
        "fingerprint": fingerprint,
        "logs": logs,
    }
    payload: dict[str, Any] = {"result": out, "meta": meta}
    out_path = art_dir / f"result_{op_key}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_operator_spec(op_dir: pathlib.Path) -> OperatorSpec:
    cfg_path = op_dir / "operator.json"
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    required = ["op_key", "op_name", "op_snake", "soc", "project_json", "pybind"]
    for k in required:
        if k not in obj:
            raise ValueError(f"missing '{k}' in {cfg_path}")
    return OperatorSpec(
        op_key=obj["op_key"],
        op_name=obj["op_name"],
        op_snake=obj["op_snake"],
        soc=obj["soc"],
        project_json=obj["project_json"],
        pybind=obj["pybind"],
    )


def import_eval_module(spec_path: pathlib.Path):
    module_name = f"llm4ascendc_eval_{spec_path.parent.parent.name}"
    m_spec = importlib.util.spec_from_file_location(module_name, str(spec_path))
    if m_spec is None or m_spec.loader is None:
        raise RuntimeError(f"failed to load eval module from {spec_path}")
    mod = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def copytree_clean(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _patch_makeself_tar_format(project_dir: pathlib.Path) -> None:
    """
    Makeself uses ustar by default, which may fail when packaging hardlink entries with long link names.
    Patch the generated makeself.cmake to pass --tar-format posix (path-related robustness).
    """
    ms = project_dir / "cmake" / "makeself.cmake"
    if not ms.exists():
        return
    txt = ms.read_text(encoding="utf-8", errors="replace")
    if "--tar-format" in txt:
        return
    needle = "execute_process(COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/util/makeself/makeself.sh"
    if needle not in txt:
        return
    patched = txt.replace(needle, needle + "\n                        --tar-format posix", 1)
    ms.write_text(patched, encoding="utf-8")


def ensure_template_pybind(pybind_dir: pathlib.Path, *, op_cpp_src: pathlib.Path, module_name: str, version: str) -> None:
    # Vendor a minimal pybind build template locally (no MKB runtime dependency).
    template_dir = TOOLS_ROOT / "pybind_template"
    setup_py = template_dir / "setup.py"
    helper_hpp = template_dir / "csrc" / "pytorch_npu_helper.hpp"
    if not setup_py.exists() or not helper_hpp.exists():
        raise FileNotFoundError(
            "pybind template missing. expected: "
            f"{setup_py} and {helper_hpp}. "
            "This will be created during migration step."
        )

    if pybind_dir.exists():
        shutil.rmtree(pybind_dir)
    (pybind_dir / "csrc").mkdir(parents=True, exist_ok=True)

    shutil.copy2(setup_py, pybind_dir / "setup.py")
    shutil.copy2(helper_hpp, pybind_dir / "csrc" / "pytorch_npu_helper.hpp")
    shutil.copy2(op_cpp_src, pybind_dir / "csrc" / "op.cpp")

    # Inject module name & version via env at build time (setup.py reads env).
    pep440_version = f"0.0.0+{version}"
    (pybind_dir / ".build_env.json").write_text(
        json.dumps({"CUSTOM_OP_NAME": module_name, "CUSTOM_OP_VERSION": pep440_version}, indent=2),
        encoding="utf-8",
    )


def build_and_install_operator(
    op_dir: pathlib.Path,
    spec: OperatorSpec,
    *,
    art_root: pathlib.Path,
    clean_policy: str,
    mode: str,
    cfg: EnvConfig,
) -> tuple[pathlib.Path, str]:
    ts = now_tag()
    art_dir = art_root / spec.op_key
    work_root = art_dir / "workspace"
    gen_dir = art_dir / "generated"
    pybind_dir = art_dir / "pybind"
    logs_dir = art_dir / "logs"
    state_dir = art_dir / "state"
    ensure = [work_root, gen_dir, pybind_dir, logs_dir, state_dir]
    for p in ensure:
        p.mkdir(parents=True, exist_ok=True)

    fp = compute_fingerprint(op_dir)
    fp_path = state_dir / "fingerprint.json"
    prev = read_fingerprint(fp_path)
    fp_changed = prev is None or prev.hexdigest != fp.hexdigest

    if clean_policy not in ("force", "smart"):
        raise ValueError("--clean-policy must be force|smart")

    if mode in ("full", "build-only"):
        if clean_policy == "force":
            if work_root.exists():
                shutil.rmtree(work_root)
            if pybind_dir.exists():
                shutil.rmtree(pybind_dir)
            work_root.mkdir(parents=True, exist_ok=True)
            pybind_dir.mkdir(parents=True, exist_ok=True)
        else:
            if fp_changed:
                if work_root.exists():
                    shutil.rmtree(work_root)
                if pybind_dir.exists():
                    shutil.rmtree(pybind_dir)
                work_root.mkdir(parents=True, exist_ok=True)
                pybind_dir.mkdir(parents=True, exist_ok=True)

    # Build step
    should_build = mode in ("full", "build-only") and (
        clean_policy == "force" or (clean_policy == "smart" and fp_changed)
    )
    if should_build:
        # 1) msopgen scaffold into artifacts workspace
        msop_json_path = work_root / f"{spec.op_snake}.json"
        msop_json_path.write_text(json.dumps(spec.project_json, ensure_ascii=False, indent=2), encoding="utf-8")
        project_dir = work_root / spec.op_name
        if project_dir.exists():
            shutil.rmtree(project_dir)

        env = build_subprocess_env(cfg)
        log_01 = logs_dir / f"{ts}-01-msopgen.log"
        run_cmd(
            ["msopgen", "gen", "-i", str(msop_json_path), "-c", spec.soc, "-lan", "cpp", "-out", spec.op_name],
            cwd=work_root,
            env=env,
            log_path=log_01,
            title=f"{spec.op_key}: msopgen",
        )

        # 2) overlay operator sources (host/kernel) into scaffold
        # Keep msopgen-generated CMakeLists.txt and other boilerplate.
        for sub in ["op_host", "op_kernel"]:
            src = op_dir / sub
            dst = project_dir / sub
            if not src.exists():
                raise FileNotFoundError(f"missing {src}")
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

        _patch_makeself_tar_format(project_dir)

        # 3) build and install .run
        log_02 = logs_dir / f"{ts}-02-build.log"
        run_cmd(
            ["bash", "build.sh"],
            cwd=project_dir,
            env=env,
            log_path=log_02,
            title=f"{spec.op_key}: build.sh",
        )
        log_03 = logs_dir / f"{ts}-03-install-run.log"
        install_cmd = ["bash", "./custom_opp_ubuntu_aarch64.run"]
        # Default installation path under /usr/local/Ascend often requires root. Use a user-writable path when configured.
        if cfg.ascend_custom_opp_path:
            pathlib.Path(cfg.ascend_custom_opp_path).mkdir(parents=True, exist_ok=True)
            # The generated .run installer is a makeself archive; in practice it is more robust to:
            # - run in quiet mode (no xterm / interaction)
            # - pass install path as --install-path=<path> (some variants don't accept a separate arg)
            install_cmd += ["--quiet", f"--install-path={cfg.ascend_custom_opp_path}"]
        run_cmd(
            install_cmd,
            cwd=project_dir / "build_out",
            env=env,
            log_path=log_03,
            title=f"{spec.op_key}: install custom_opp run",
        )

        # 4) keep a readable copy of generated scaffold + sources
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        for rel in ["op_host", "op_kernel", "framework", "cmake", "CMakeLists.txt", "CMakePresets.json", "build.sh"]:
            s = project_dir / rel
            d = gen_dir / rel
            if s.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True)
            elif s.is_file():
                shutil.copy2(s, d)

        # 5) build & install pybind wheel (force reinstall)
        module_name = spec.pybind.get("module_name", "custom_ops_lib")
        op_cpp_src = op_dir / "pybind" / "op.cpp"
        if not op_cpp_src.exists():
            raise FileNotFoundError(f"missing {op_cpp_src}")
        ensure_template_pybind(pybind_dir, op_cpp_src=op_cpp_src, module_name=module_name, version=fp.short())

        build_env_path = pybind_dir / ".build_env.json"
        build_env_obj = json.loads(build_env_path.read_text(encoding="utf-8"))
        env2 = env.copy()
        env2.update({k: str(v) for k, v in build_env_obj.items()})

        prefix = shell_prefix(cfg)
        log_04 = logs_dir / f"{ts}-04-pybind-build.log"
        run_cmd(
            ["bash", "-lc", f"{prefix} && python3 setup.py build bdist_wheel"],
            cwd=pybind_dir,
            env=env2,
            log_path=log_04,
            title=f"{spec.op_key}: pybind wheel build",
        )
        wheels = sorted((pybind_dir / "dist").glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"{spec.op_key}: wheel not generated under {pybind_dir/'dist'}")
        wheel_path = wheels[-1]
        log_05 = logs_dir / f"{ts}-05-pybind-install.log"
        run_cmd(
            ["bash", "-lc", f"{prefix} && pip3 install '{wheel_path}' --force-reinstall"],
            cwd=pybind_dir,
            env=env2,
            log_path=log_05,
            title=f"{spec.op_key}: pip install wheel",
        )

        # 6) record fingerprint + installed wheel info
        write_fingerprint(fp_path, fp)
        (state_dir / "installed.json").write_text(
            json.dumps({"module_name": module_name, "wheel": str(wheel_path), "fingerprint": fp.hexdigest}, indent=2),
            encoding="utf-8",
        )

    # smart policy: if no changes and build-only/full, we still want to validate installed state exists
    installed = (
        json.loads((state_dir / "installed.json").read_text(encoding="utf-8"))
        if (state_dir / "installed.json").exists()
        else {}
    )
    if mode in ("full", "build-only") and clean_policy == "smart" and (not should_build) and not installed:
        raise RuntimeError(f"{spec.op_key}: smart mode wants reuse but no installed state; run with --clean-policy force")
    return art_dir, installed.get("module_name", spec.pybind.get("module_name", "custom_ops_lib"))


def run_eval(op_dir: pathlib.Path, spec: OperatorSpec, *, module_name: str, art_dir: pathlib.Path, cfg: EnvConfig) -> int:
    eval_spec_path = op_dir / "eval" / "spec.py"
    if not eval_spec_path.exists():
        raise FileNotFoundError(f"missing {eval_spec_path}")

    logs_dir = art_dir / "logs"
    ts = now_tag()
    log_path = logs_dir / f"{ts}-06-eval.log"

    # Execute eval as a separate python process to ensure fresh import state.
    env = build_subprocess_env(cfg)
    env["LLM4ASCENDC_OP_MODULE"] = module_name
    env["LLM4ASCENDC_OP_DIR"] = str(op_dir)
    env["LLM4ASCENDC_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH','')}"

    # Ensure Ascend env + conda in shell context when running python.
    prefix = shell_prefix(cfg)
    cmd = ["bash", "-lc", f"{prefix} && python3 '{eval_spec_path}'"]
    run_cmd(cmd, cwd=ROOT, env=env, log_path=log_path, title=f"{spec.op_key}: eval")
    return 0


def _resolve_user_path(p: pathlib.Path) -> pathlib.Path:
    if not p.is_absolute():
        return (pathlib.Path.cwd() / p).resolve()
    return p.resolve()


def _is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _artifact_group_rel_from_txt_path(txt_path: pathlib.Path) -> pathlib.Path | None:
    """
    If txt_path is under LLM4AscendC/output/<group>/..., return the group path
    relative to output root (mirrors nested folders).

    For a single txt file, the group is its parent directory relative to output root.
    If the parent is output root itself, returns None (treat as 'wild').
    """
    txt_path = txt_path.resolve()
    out_root = OUTPUT_ROOT.resolve()
    if not _is_within(txt_path, out_root):
        return None
    try:
        rel_parent = txt_path.parent.relative_to(out_root)
    except Exception:
        return None
    return rel_parent if str(rel_parent) not in (".", "") else None


def _artifact_group_rel_from_txt_dir(txt_dir: pathlib.Path) -> pathlib.Path | None:
    """
    If txt_dir is under LLM4AscendC/output/<group>/..., return the directory path
    relative to output root (mirrors nested folders).
    """
    txt_dir = txt_dir.resolve()
    out_root = OUTPUT_ROOT.resolve()
    if not _is_within(txt_dir, out_root):
        return None
    try:
        rel = txt_dir.relative_to(out_root)
    except Exception:
        return None
    return rel if str(rel) not in (".", "") else None


def _artifacts_root_for_group(group_rel: pathlib.Path | None) -> pathlib.Path:
    return (ART_ROOT / group_rel) if group_rel else ART_ROOT


def _execute_pipeline(op_dir: pathlib.Path, *, art_root: pathlib.Path, mode: str, clean_policy: str) -> None:
    """
    Build/install/eval one operator directory. Raises on failure.
    """
    spec = load_operator_spec(op_dir)
    cfg = load_env_config()
    art_dir = art_root / spec.op_key
    state_dir = art_dir / "state"
    logs_dir = art_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    fp_now = compute_fingerprint(op_dir)
    fp_now_hex = fp_now.hexdigest

    logs: dict[str, str] = {}
    compiled_ok: bool | None = None
    correctness_ok: bool | None = None
    correctness_info: str | None = None

    try:
        if mode in ("full", "build-only"):
            _, module_name = build_and_install_operator(
                op_dir, spec, art_root=art_root, clean_policy=clean_policy, mode=mode, cfg=cfg
            )
            compiled_ok = True
            for k in ["01-msopgen", "02-build", "03-install-run", "04-pybind-build", "05-pybind-install"]:
                latest = sorted(logs_dir.glob(f"*-{k}.log"))
                if latest:
                    logs[k] = str(latest[-1])
            if mode == "build-only":
                out_path = write_result_json(
                    art_dir=art_dir,
                    op_key=spec.op_key,
                    compiled=compiled_ok,
                    correctness=None,
                    correctness_info=None,
                    logs=logs,
                    fingerprint=fp_now_hex,
                    mode=mode,
                )
                print(f"[done] build-only OK. summary: {out_path}")
                return

        else:
            state = state_dir / "installed.json"
            if not state.exists():
                raise RuntimeError(f"no installed state for {spec.op_key}. run build first.")
            installed = json.loads(state.read_text(encoding="utf-8"))
            module_name = installed["module_name"]

            prev_fp = read_fingerprint(state_dir / "fingerprint.json")
            if prev_fp is None or prev_fp.hexdigest != fp_now_hex:
                raise RuntimeError(
                    f"{spec.op_key}: sources changed since last build; refuse eval-only. "
                    "Run with --mode full or build-only first."
                )
            compiled_ok = True

        if mode in ("full", "eval-only"):
            run_eval(op_dir, spec, module_name=module_name, art_dir=art_dir, cfg=cfg)
            latest_eval = sorted(logs_dir.glob("*-06-eval.log"))
            if latest_eval:
                eval_log = latest_eval[-1]
                logs["06-eval"] = str(eval_log)
                tail = _read_tail(eval_log, max_lines=120)
                if "[smoke] no model_src" in tail or "[smoke] no eval_src provided" in tail:
                    correctness_ok = None
                    correctness_info = "No model_src/eval_src (smoke test only)"
                else:
                    correctness_ok = True
            else:
                correctness_ok = True

        out_path = write_result_json(
            art_dir=art_dir,
            op_key=spec.op_key,
            compiled=compiled_ok,
            correctness=correctness_ok,
            correctness_info=correctness_info,
            logs=logs,
            fingerprint=fp_now_hex,
            mode=mode,
        )
        print(f"[done] OK. summary: {out_path}")

    except Exception as e:
        compiled_ok = compiled_ok if compiled_ok is not None else False
        correctness_ok = correctness_ok if correctness_ok is not None else False

        newest_logs = sorted(logs_dir.glob("*.log"))
        newest = newest_logs[-1] if newest_logs else None
        tail = _read_tail(newest) if newest else ""
        core = _extract_core_error(tail)
        correctness_info = core or f"{type(e).__name__}: {e}"

        for k in ["01-msopgen", "02-build", "03-install-run", "04-pybind-build", "05-pybind-install", "06-eval"]:
            latest = sorted(logs_dir.glob(f"*-{k}.log"))
            if latest:
                logs[k] = str(latest[-1])

        out_path = write_result_json(
            art_dir=art_dir,
            op_key=spec.op_key,
            compiled=compiled_ok,
            correctness=correctness_ok if mode != "build-only" else None,
            correctness_info=correctness_info,
            logs=logs,
            fingerprint=fp_now_hex,
            mode=mode,
        )
        print(f"[done] FAILED. summary: {out_path}")
        raise


def _parallel_worker_main(
    worker_id: int,
    base_opp: str,
    task_q: Any,
    art_root_str: str,
    mode: str,
    clean_policy: str,
    result_q: Any,
) -> None:
    """
    Runs in a child process (multiprocessing spawn). Each worker uses a fixed
    OPP install root: <base_opp>/_parallel_w<worker_id> via LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH.
    Consumes txt paths from task_q until sentinel None.
    """
    opp = pathlib.Path(base_opp) / f"_parallel_w{worker_id}"
    opp.mkdir(parents=True, exist_ok=True)
    os.environ["LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH"] = str(opp)
    art_root = pathlib.Path(art_root_str)
    while True:
        item = task_q.get()
        if item is None:
            break
        txt_path = pathlib.Path(item)
        print(f"[batch] [w{worker_id}] === {txt_path.name} ===")
        staging_root = art_root / "_txt_staging" / txt_path.stem
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            op_dir = materialize_operator_from_txt(
                out_dir=staging_root / "operator",
                txt_path=txt_path,
                soc="ai_core-Ascend910B2",
            )
            _execute_pipeline(op_dir, art_root=art_root, mode=mode, clean_policy=clean_policy)
            result_q.put((txt_path.name, True, None))
        except Exception as e:
            print(f"[batch] FAILED {txt_path.name}: {type(e).__name__}: {e}")
            result_q.put((txt_path.name, False, str(e)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Unified AscendC operator build/eval. "
            "MKB-style txt bundles: use output/<op_key>.txt where op_key matches "
            "vendor/mkb/dataset.py (filename stem is the MKB op key)."
        )
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--op", help="operator directory (e.g. operators/add)")
    src.add_argument(
        "--txt",
        help="single txt bundle path (MKB-style blocks; basename stem must equal MKB op_key, e.g. output/layer_norm.txt)",
    )
    src.add_argument(
        "--txt-dir",
        dest="txt_dir",
        metavar="DIR",
        help="directory of *.txt bundles (each file stem = MKB op_key); runs all files sequentially",
    )
    ap.add_argument("--mode", default="full", choices=["full", "build-only", "eval-only"])
    ap.add_argument("--clean-policy", default="force", choices=["force", "smart"])
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "only with --txt-dir: number of parallel worker processes (default 1 = sequential). "
            f"Requires LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH; uses subdirs _parallel_w0.. under that path. "
            f"Max {_MAX_TXT_DIR_WORKERS}."
        ),
    )
    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > _MAX_TXT_DIR_WORKERS:
        raise ValueError(f"--workers must be <= {_MAX_TXT_DIR_WORKERS}")
    if args.workers > 1 and args.txt_dir is None:
        raise ValueError("--workers > 1 is only supported with --txt-dir")

    if args.txt_dir:
        txt_dir = _resolve_user_path(pathlib.Path(args.txt_dir))
        if not txt_dir.is_dir():
            raise NotADirectoryError(txt_dir)
        group_rel = _artifact_group_rel_from_txt_dir(txt_dir)
        art_root = _artifacts_root_for_group(group_rel)
        files = sorted(txt_dir.glob("*.txt"))
        if not files:
            raise RuntimeError(f"no .txt files in {txt_dir}")

        if args.workers == 1:
            rc = 0
            for txt_path in files:
                print(f"[batch] === {txt_path.name} ===")
                staging_root = art_root / "_txt_staging" / txt_path.stem
                if staging_root.exists():
                    shutil.rmtree(staging_root)
                staging_root.mkdir(parents=True, exist_ok=True)
                op_dir = materialize_operator_from_txt(
                    out_dir=staging_root / "operator",
                    txt_path=txt_path,
                    soc="ai_core-Ascend910B2",
                )
                try:
                    _execute_pipeline(op_dir, art_root=art_root, mode=args.mode, clean_policy=args.clean_policy)
                except Exception as e:
                    print(f"[batch] FAILED {txt_path.name}: {type(e).__name__}: {e}")
                    rc = 1
            return rc

        cfg_parallel = load_env_config()
        if not cfg_parallel.ascend_custom_opp_path:
            raise RuntimeError(
                "Parallel --txt-dir (--workers > 1) requires a writable OPP root. "
                "Set environment variable LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH to an absolute path "
                "(see tools/common/env.py). Each worker installs to <that path>/_parallel_w<id>."
            )
        base_opp = cfg_parallel.ascend_custom_opp_path
        ctx = multiprocessing.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        for f in files:
            task_q.put(str(f.resolve()))
        for _ in range(args.workers):
            task_q.put(None)
        procs: list[multiprocessing.Process] = []
        art_root_str = str(art_root.resolve())
        for wid in range(args.workers):
            p = ctx.Process(
                target=_parallel_worker_main,
                args=(wid, base_opp, task_q, art_root_str, args.mode, args.clean_policy, result_q),
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

    if args.op:
        op_dir = pathlib.Path(args.op)
        if not op_dir.is_absolute():
            op_dir = (ROOT / op_dir).resolve()
        if not op_dir.exists():
            raise FileNotFoundError(op_dir)
    else:
        txt_path = _resolve_user_path(pathlib.Path(args.txt))
        if not txt_path.exists():
            raise FileNotFoundError(txt_path)
        group_rel = _artifact_group_rel_from_txt_path(txt_path)
        art_root = _artifacts_root_for_group(group_rel)
        staging_root = art_root / "_txt_staging" / txt_path.stem
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        op_dir = materialize_operator_from_txt(
            out_dir=staging_root / "operator",
            txt_path=txt_path,
            soc="ai_core-Ascend910B2",
        )

    if args.op:
        art_root = ART_ROOT
    _execute_pipeline(op_dir, art_root=art_root, mode=args.mode, clean_policy=args.clean_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

