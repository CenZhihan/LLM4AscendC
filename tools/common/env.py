#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    conda_env: str | None = "multi-kernel-bench"
    ascend_set_env: str = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    driver_libs: tuple[str, str] = ("/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64/common")
    # Install custom OPP packages to a user-writable path and expose it via ASCEND_CUSTOM_OPP_PATH.
    # Keep it None to disable.
    ascend_custom_opp_path: str | None = "/workspace/ascend_custom_opp"


_ASCEND_CUSTOM_OPP_ENV = "LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH"


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
    ld = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    for p in cfg.driver_libs:
        if p not in parts:
            parts.insert(0, p)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    if cfg.ascend_custom_opp_path:
        env["ASCEND_CUSTOM_OPP_PATH"] = cfg.ascend_custom_opp_path
    return env


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
        conda_sh = "/root/miniconda3/etc/profile.d/conda.sh"
        if os.path.exists(conda_sh):
            pieces.append(f"source '{conda_sh}' && conda activate {cfg.conda_env}")
    return " && ".join(pieces)

