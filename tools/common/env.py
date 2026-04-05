#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# 项目根目录（env.py 在 tools/common/ 下，往上两层是根目录）
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EnvConfig:
    conda_env: str | None = "multi-kernel-bench"
    ascend_set_env: str = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    driver_libs: tuple[str, str] = ("/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64/common")
    # Install custom OPP packages to a user-writable path and expose it via ASCEND_CUSTOM_OPP_PATH.
    # Keep it None to disable.
    # 默认使用项目根目录下的 ascend_custom_opp，而非 Docker 容器内的 /workspace
    ascend_custom_opp_path: str | None = str(ROOT_DIR / "ascend_custom_opp")


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

