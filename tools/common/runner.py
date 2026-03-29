#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    returncode: int
    log_path: pathlib.Path


def now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    log_path: pathlib.Path,
    title: str,
) -> RunResult:
    ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"[run] {title}")
    print(f"[run] cwd={cwd}")
    print(f"[run] cmd={' '.join(cmd)}")
    print(f"[run] log={log_path}")
    sys.stdout.flush()

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# title: {title}\n")
        f.write(f"# cwd: {cwd}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"\n# returncode: {rc}\n")

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, f"see log: {log_path}")
    return RunResult(returncode=rc, log_path=log_path)

