#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import os
import pathlib
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

DEFAULT_EVAL_TIMEOUT_SEC = 1200
EVAL_TIMEOUT_ENV = "LLM4ASCENDC_EVAL_TIMEOUT_SEC"
# Align with GNU timeout(1) exit code for agent synthetic JSON.
EXIT_CODE_EVAL_TIMEOUT = 124


@dataclass(frozen=True)
class RunResult:
    returncode: int
    log_path: pathlib.Path


class CommandTimeoutError(TimeoutError):
    def __init__(self, *, title: str, timeout_sec: float, log_path: pathlib.Path) -> None:
        self.title = title
        self.timeout_sec = timeout_sec
        self.log_path = log_path
        super().__init__(
            f"command timed out after {timeout_sec}s ({title}); see log: {log_path}"
        )


def now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_eval_timeout_sec(cli_value: int | None = None) -> float | None:
    """
    Resolve NPU eval wall-clock timeout (seconds).

    Priority: CLI --eval-timeout > LLM4ASCENDC_EVAL_TIMEOUT_SEC > DEFAULT (1200).
    Values <= 0 disable the timeout (debug only).
    """
    if cli_value is not None:
        raw = cli_value
    else:
        env_raw = os.environ.get(EVAL_TIMEOUT_ENV, "").strip()
        if env_raw:
            try:
                raw = int(env_raw)
            except ValueError:
                raw = DEFAULT_EVAL_TIMEOUT_SEC
        else:
            raw = DEFAULT_EVAL_TIMEOUT_SEC
    if raw <= 0:
        return None
    return float(raw)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _drain_stdout(proc: subprocess.Popen[str], f, out_write) -> None:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line == "":
            break
        out_write(line)


def _run_with_timeout(
    proc: subprocess.Popen[str],
    f,
    *,
    timeout_sec: float,
    log_path: pathlib.Path,
    title: str,
) -> int:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_sec

    def out_write(line: str) -> None:
        sys.stdout.write(line)
        f.write(line)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(proc)
            f.write(f"\n# TIMEOUT after {timeout_sec}s\n")
            f.flush()
            raise CommandTimeoutError(
                title=title, timeout_sec=timeout_sec, log_path=log_path
            )

        ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.25))
        if ready:
            line = proc.stdout.readline()
            if line:
                out_write(line)
                continue
            if proc.poll() is not None:
                break
        elif proc.poll() is not None:
            _drain_stdout(proc, f, out_write)
            break

    return int(proc.wait())


def run_cmd(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    log_path: pathlib.Path,
    title: str,
    timeout_sec: float | None = None,
) -> RunResult:
    ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    print(f"[run] {title}")
    print(f"[run] cwd={cwd}")
    print(f"[run] cmd={' '.join(cmd)}")
    print(f"[run] log={log_path}")
    if timeout_sec is not None:
        print(f"[run] eval_timeout_sec={timeout_sec}")
    sys.stdout.flush()

    use_timeout = timeout_sec is not None and timeout_sec > 0
    popen_kw: dict = {
        "cwd": str(cwd),
        "env": merged_env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "universal_newlines": True,
    }
    if use_timeout:
        popen_kw["start_new_session"] = True

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# title: {title}\n")
        f.write(f"# cwd: {cwd}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n")
        if use_timeout:
            f.write(f"# eval_timeout_sec: {timeout_sec}\n")
        f.write("\n")
        f.flush()

        proc = subprocess.Popen(cmd, **popen_kw)
        assert proc.stdout is not None
        try:
            if use_timeout:
                rc = _run_with_timeout(
                    proc, f, timeout_sec=float(timeout_sec), log_path=log_path, title=title
                )
            else:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    f.write(line)
                rc = proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        f.write(f"\n# returncode: {rc}\n")

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, f"see log: {log_path}")
    return RunResult(returncode=rc, log_path=log_path)
