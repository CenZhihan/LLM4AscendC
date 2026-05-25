#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# 项目根目录（env.py 在 tools/common/ 下，往上两层是根目录）
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EnvConfig:
    conda_env: str | None = "czh_environ"
    ascend_set_env: str = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    driver_libs: tuple[str, str] = ("/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64/common")
    # Install custom OPP packages to a user-writable path and expose it via ASCEND_CUSTOM_OPP_PATH.
    # Keep it None to disable.
    # 默认使用项目根目录下的 ascend_custom_opp，而非 Docker 容器内的 /workspace
    ascend_custom_opp_path: str | None = str(ROOT_DIR / "ascend_custom_opp")


_ASCEND_CUSTOM_OPP_ENV = "LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH"
_ASCEND_CUSTOM_OPP_BASE_ENV = "LLM4ASCENDC_ASCEND_CUSTOM_OPP_BASE"
_PARALLEL_W_DIR_RE = re.compile(r"^_parallel_w\d+$")
# CANN 8.x TBE/AscendC 编译链绑定 Python 3.11；conda 评测环境多为 3.10（torch cp310）。
# 若 build.sh 误用 3.10，常见现象为 kernel *.json 为空、JSONDecodeError(EB0500)。
_CANN_BUILD_PYTHON_CANDIDATES = (
    "/usr/local/python3.11.13/bin",
    "/usr/local/bin",
)


def _prepend_cann_build_python(env: dict[str, str]) -> None:
    if env.get("ASCEND_PYTHON_EXECUTABLE", "").strip():
        py = Path(env["ASCEND_PYTHON_EXECUTABLE"])
        if py.is_file():
            bin_dir = str(py.parent)
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        return
    for bin_dir in _CANN_BUILD_PYTHON_CANDIDATES:
        py = Path(bin_dir) / "python3"
        if not py.is_file():
            continue
        try:
            out = subprocess.check_output(
                [str(py), "-c", "import sys; print(sys.version_info[:2])"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if out.replace(" ", "") != "(3,11)":
            continue
        env["ASCEND_PYTHON_EXECUTABLE"] = str(py)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        _strip_conda_from_pythonpath(env)
        return


def _parse_bash_export_p(env: dict[str, str], export_p_text: str) -> None:
    """Merge `bash -c 'source ... && export -p'` output into env."""
    for line in export_p_text.splitlines():
        if line.startswith("declare -x "):
            body = line[len("declare -x ") :]
        elif line.startswith("export "):
            body = line[len("export ") :]
        else:
            continue
        if "=" not in body:
            continue
        key, _, val = body.partition("=")
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        env[key] = val


def _ingest_ascend_set_env(cfg: EnvConfig, env: dict[str, str]) -> None:
    """Ensure CANN toolkit variables exist (spawn workers may miss parent `source`)."""
    if not os.path.exists(cfg.ascend_set_env):
        return
    try:
        out = subprocess.check_output(
            ["bash", "-c", f"source '{cfg.ascend_set_env}' >/dev/null 2>&1 && export -p"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    _parse_bash_export_p(env, out)


def _apply_build_job_limits(env: dict[str, str]) -> None:
    """
    Cap cmake/make parallelism per eval subprocess.
    With --workers>1, each worker runs build.sh which uses -j$(nproc); on large hosts
    this can oversubscribe and trigger flaky TBE JSON / binary/config failures.
    """
    jobs = os.environ.get("LLM4ASCENDC_BUILD_JOBS", "16").strip() or "16"
    env["MAKEFLAGS"] = f"-j{jobs}"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = jobs


def _strip_conda_from_pythonpath(env: dict[str, str]) -> None:
    """避免 CANN TBE 编译时 PYTHONPATH 混入 conda(3.10) site-packages。"""
    pp = env.get("PYTHONPATH", "")
    if not pp:
        return
    parts = [
        p
        for p in pp.split(":")
        if p and "miniconda" not in p and "/conda/" not in p and "envs/" not in p
    ]
    env["PYTHONPATH"] = ":".join(parts)


def resolve_ascend_custom_opp_base(path: str | None = None) -> str:
    """
    Canonical OPP install root without per-worker suffixes.

    Agent/eval may set LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH to .../_parallel_w2; calling
    init_parallel_* again must not append another _parallel_w* on top (nested paths).
    """
    explicit_base = os.environ.get(_ASCEND_CUSTOM_OPP_BASE_ENV, "").strip()
    if path is None and explicit_base:
        return str(Path(explicit_base).resolve())

    raw = (path or os.environ.get(_ASCEND_CUSTOM_OPP_ENV, "")).strip()
    if not raw:
        return str((ROOT_DIR / "ascend_custom_opp").resolve())

    cur = Path(raw).resolve()
    while _PARALLEL_W_DIR_RE.fullmatch(cur.name):
        cur = cur.parent
    return str(cur)


def parallel_opp_path_for_bucket(*, base_opp: str, bucket: int) -> str:
    base = resolve_ascend_custom_opp_base(base_opp)
    return str(Path(base) / f"_parallel_w{bucket}")


def load_env_config() -> EnvConfig:
    """
    若设置 LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH，则覆盖默认 ascend_custom_opp_path。
    用于集群任务：容器内常无权限写 /workspace，可指向共享盘下目录（与调试机侧载 /workspace 时默认路径不同）。
    """
    override = os.environ.get(_ASCEND_CUSTOM_OPP_ENV, "").strip()
    if not override:
        return EnvConfig()
    return EnvConfig(ascend_custom_opp_path=override)


def build_subprocess_env(cfg: EnvConfig) -> dict[str, str]:
    env = os.environ.copy()
    _ingest_ascend_set_env(cfg, env)
    ld = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    for p in cfg.driver_libs:
        if p not in parts:
            parts.insert(0, p)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    if cfg.ascend_custom_opp_path:
        env["ASCEND_CUSTOM_OPP_PATH"] = cfg.ascend_custom_opp_path
    _prepend_cann_build_python(env)
    _apply_build_job_limits(env)
    return env


def ensure_parallel_build_jobs(*, worker_count: int) -> str:
    """
    Set LLM4ASCENDC_BUILD_JOBS when unset (cap per-worker compile threads).
    Returns the jobs value in effect.
    """
    existing = os.environ.get("LLM4ASCENDC_BUILD_JOBS", "").strip()
    if existing:
        return existing
    ncpu = os.cpu_count() or 16
    per_worker = max(4, min(16, ncpu // max(1, worker_count)))
    os.environ["LLM4ASCENDC_BUILD_JOBS"] = str(per_worker)
    return str(per_worker)


def apply_agent_parallel_slot_env(*, op_slot: int, parallel_ops: int, npu_count: int) -> None:
    """
    ProcessPoolExecutor child initializer for run_agent_* multi-round scripts.
    """
    if parallel_ops <= 1:
        device_id = op_slot % max(1, npu_count)
        os.environ["ASCEND_VISIBLE_DEVICES"] = str(device_id)
        os.environ.update(build_subprocess_env(load_env_config()))
        return
    init_parallel_op_slot_os_environ(
        op_slot=op_slot,
        parallel_ops=parallel_ops,
        npu_count=npu_count,
        label="agent",
    )


def init_parallel_op_slot_os_environ(
    *,
    op_slot: int,
    parallel_ops: int,
    npu_count: int,
    label: str = "op",
) -> EnvConfig:
    """
  ProcessPoolExecutor / agent parallel row jobs: isolate OPP by slot bucket and bind NPU.

  - OPP root: <base>/_parallel_w{op_slot % parallel_ops}
  - NPU: ASCEND_VISIBLE_DEVICES = op_slot % npu_count
    """
    parallel_ops = max(1, parallel_ops)
    npu_count = max(1, npu_count)
    opp_bucket = op_slot % parallel_ops
    device_id = op_slot % npu_count
    base_opp = resolve_ascend_custom_opp_base()
    if parallel_ops > 1:
        opp = parallel_opp_path_for_bucket(base_opp=base_opp, bucket=opp_bucket)
        Path(opp).mkdir(parents=True, exist_ok=True)
        os.environ[_ASCEND_CUSTOM_OPP_BASE_ENV] = base_opp
        os.environ[_ASCEND_CUSTOM_OPP_ENV] = opp
    else:
        opp = base_opp
        os.environ[_ASCEND_CUSTOM_OPP_BASE_ENV] = base_opp
        os.environ[_ASCEND_CUSTOM_OPP_ENV] = base_opp
    os.environ["ASCEND_VISIBLE_DEVICES"] = str(device_id)
    merged = build_subprocess_env(load_env_config())
    os.environ.update(merged)
    print(
        f"[batch] [{label} slot={op_slot}] ASCEND_VISIBLE_DEVICES={device_id}, "
        f"OPP={opp}, BUILD_JOBS={merged.get('CMAKE_BUILD_PARALLEL_LEVEL', '?')}, "
        f"ASCEND_PYTHON={merged.get('ASCEND_PYTHON_EXECUTABLE', '(default)')}"
    )
    return load_env_config()


def init_parallel_worker_os_environ(
    *,
    worker_id: int,
    base_opp: str,
    npu_count: int,
) -> EnvConfig:
    """
    Called at the start of each eval_operator multiprocessing spawn worker.
    Binds NPU, isolates OPP install root, and merges CANN build env into os.environ.
    """
    device_id = worker_id % max(1, npu_count)
    base = resolve_ascend_custom_opp_base(base_opp)
    opp = parallel_opp_path_for_bucket(base_opp=base, bucket=worker_id)
    Path(opp).mkdir(parents=True, exist_ok=True)
    os.environ["ASCEND_VISIBLE_DEVICES"] = str(device_id)
    os.environ[_ASCEND_CUSTOM_OPP_BASE_ENV] = base
    os.environ[_ASCEND_CUSTOM_OPP_ENV] = opp
    merged = build_subprocess_env(load_env_config())
    os.environ.update(merged)
    print(
        f"[batch] [w{worker_id}] env: ASCEND_VISIBLE_DEVICES={device_id}, "
        f"OPP={opp}, BUILD_JOBS={merged.get('CMAKE_BUILD_PARALLEL_LEVEL', '?')}, "
        f"ASCEND_PYTHON={merged.get('ASCEND_PYTHON_EXECUTABLE', '(default)')}"
    )
    return load_env_config()


def shell_prefix(cfg: EnvConfig) -> str:
    pieces: list[str] = []
    if os.path.exists(cfg.ascend_set_env):
        pieces.append(f"source '{cfg.ascend_set_env}'")
    # If we installed custom OPP into a user-writable location, its set_env.bash
    # is usually required to make libopapi.so discoverable at runtime.
    if cfg.ascend_custom_opp_path:
        custom_set_env = os.path.join(cfg.ascend_custom_opp_path, "vendors", "customize", "bin", "set_env.bash")
        if os.path.exists(custom_set_env):
            pieces.append(f"source '{custom_set_env}'")
    if cfg.conda_env:
        conda_sh = str(ROOT_DIR / "miniconda3" / "etc" / "profile.d" / "conda.sh")
        if os.path.exists(conda_sh):
            env_bin = str(ROOT_DIR / "miniconda3" / "envs" / cfg.conda_env / "bin")
            pieces.append(
                f"source '{conda_sh}' && conda activate {cfg.conda_env} && "
                f"export PATH=\"{env_bin}:$PATH\""
            )
    return " && ".join(pieces)

