from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generator.scripts.run_agent_two_rounds import (
    _attempt_failed,
    _build_repair_error_context,
    _select_error_logs,
)


class TestTwoRoundRunnerHelpers(unittest.TestCase):
    def test_select_error_logs_prefers_build_and_eval(self):
        logs = {
            "01-msopgen": "/tmp/01.log",
            "02-build": "/tmp/02.log",
            "06-eval": "/tmp/06.log",
            "04-pybind-build": "/tmp/04.log",
        }
        selected = _select_error_logs(logs)
        self.assertEqual(selected, ["/tmp/02.log", "/tmp/06.log"])

    def test_select_error_logs_falls_back_when_no_build_eval(self):
        logs = {
            "01-msopgen": "/tmp/01.log",
            "04-pybind-build": "/tmp/04.log",
        }
        selected = _select_error_logs(logs)
        self.assertEqual(selected, ["/tmp/01.log", "/tmp/04.log"])

    def test_build_repair_context_keeps_raw_text_sections(self):
        with tempfile.TemporaryDirectory() as td:
            build_log = Path(td) / "build.log"
            eval_log = Path(td) / "eval.log"
            build_log.write_text("line1\nline2\ncompile error x\n", encoding="utf-8")
            eval_log.write_text("runtime fail y\n", encoding="utf-8")
            payload = {
                "result": {
                    "softmax": {
                        "compiled": False,
                        "correctness": False,
                        "correctness_info": "raw correctness info",
                    }
                }
            }
            ctx = _build_repair_error_context(
                op="softmax",
                result_payload=payload,
                selected_logs=[str(build_log), str(eval_log)],
                max_log_lines=80,
            )
            self.assertIn("raw correctness info", ctx)
            self.assertIn("compile error x", ctx)
            self.assertIn("runtime fail y", ctx)
            self.assertIn("log file:", ctx)

    def test_attempt_failed_rule(self):
        self.assertTrue(_attempt_failed(False, False))
        self.assertTrue(_attempt_failed(True, False))
        self.assertFalse(_attempt_failed(True, True))


if __name__ == "__main__":
    unittest.main()
