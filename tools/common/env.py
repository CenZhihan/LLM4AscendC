#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    conda_env: str | None = "multi-kernel-bench"
    ascend_set_env: str = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    driver_libs: tuple[str, str] = ("/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64/common")


def build_subprocess_env(cfg: EnvConfig) -> dict[str, str]:
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    for p in cfg.driver_libs:
        if p not in parts:
            parts.insert(0, p)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def shell_prefix(cfg: EnvConfig) -> str:
    pieces: list[str] = []
    if os.path.exists(cfg.ascend_set_env):
        pieces.append(f"source '{cfg.ascend_set_env}'")
    if cfg.conda_env:
        conda_sh = "/root/miniconda3/etc/profile.d/conda.sh"
        if os.path.exists(conda_sh):
            pieces.append(f"source '{conda_sh}' && conda activate {cfg.conda_env}")
    return " && ".join(pieces)

