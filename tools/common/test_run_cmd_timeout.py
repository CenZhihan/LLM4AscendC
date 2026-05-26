#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from tools.common.runner import (
    CommandTimeoutError,
    resolve_eval_timeout_sec,
    run_cmd,
)


class TestResolveEvalTimeoutSec(unittest.TestCase):
    def test_default_when_unset(self) -> None:
        env = os.environ.copy()
        env.pop("LLM4ASCENDC_EVAL_TIMEOUT_SEC", None)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(resolve_eval_timeout_sec(None), 1200.0)

    def test_cli_overrides_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"LLM4ASCENDC_EVAL_TIMEOUT_SEC": "99"}):
            self.assertEqual(resolve_eval_timeout_sec(30), 30.0)

    def test_zero_disables(self) -> None:
        self.assertIsNone(resolve_eval_timeout_sec(0))


class TestRunCmdTimeout(unittest.TestCase):
    def test_sleep_command_times_out_and_kills_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "timeout.log"
            t0 = time.monotonic()
            with self.assertRaises(CommandTimeoutError) as ctx:
                run_cmd(
                    ["bash", "-c", "sleep 30"],
                    cwd=Path(tmp),
                    log_path=log_path,
                    title="test: eval",
                    timeout_sec=1.0,
                )
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 8.0)
            self.assertIn("TIMEOUT", log_path.read_text(encoding="utf-8"))
            self.assertEqual(ctx.exception.timeout_sec, 1.0)

            probe = subprocess.run(
                ["pgrep", "-f", "sleep 30"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(probe.returncode, 0, msg="sleep 30 should not remain running")


if __name__ == "__main__":
    unittest.main()
