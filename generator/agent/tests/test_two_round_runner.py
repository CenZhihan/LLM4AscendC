from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generator.scripts.run_agent_multi_rounds import (
    _attempt_failed,
    _build_repair_error_context,
    _run_eval_for_txt,
    _select_error_logs,
    run_multi_attempt_for_op,
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

    def test_run_eval_for_txt_rejects_workers_gt_one(self):
        with self.assertRaises(ValueError):
            _run_eval_for_txt(Path("/tmp/a.txt"), mode="full", clean_policy="force", eval_workers=2)

    def test_run_multi_attempt_supports_three_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            result_paths = {}
            for idx in (1, 2, 3):
                p = run_dir / f"result_attempt{idx}.json"
                p.write_text("{}", encoding="utf-8")
                result_paths[idx] = p

            payloads = [
                {"result": {"elu": {"compiled": False, "correctness": False, "correctness_info": "a1"}}, "meta": {"logs": {}}},
                {"result": {"elu": {"compiled": False, "correctness": False, "correctness_info": "a2"}}, "meta": {"logs": {}}},
                {"result": {"elu": {"compiled": True, "correctness": True, "correctness_info": ""}}, "meta": {"logs": {}}},
            ]
            payload_idx = {"i": 0}

            def fake_generate(**kwargs):
                attempt_id = kwargs["attempt_id"]
                out_dir = kwargs["out_dir"]
                txt_path = out_dir / f"elu_a{attempt_id}.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text("code", encoding="utf-8")
                return {
                    "result": SimpleNamespace(generated_code=f"code_a{attempt_id}"),
                    "txt_path": txt_path,
                    "report_path": out_dir / f"elu_a{attempt_id}_report.json",
                }

            def fake_eval_result_path(txt_path, op):
                _ = op
                attempt_id = int(txt_path.parent.name.replace("attempt", ""))
                return result_paths[attempt_id]

            def fake_load_payload(_path):
                i = payload_idx["i"]
                payload_idx["i"] = i + 1
                return payloads[i]

            with patch("generator.scripts.run_agent_multi_rounds._generate_one_attempt", side_effect=fake_generate), \
                 patch("generator.scripts.run_agent_multi_rounds._run_eval_for_txt", return_value=0), \
                 patch("generator.scripts.run_agent_multi_rounds._eval_result_json_path", side_effect=fake_eval_result_path), \
                 patch("generator.scripts.run_agent_multi_rounds._load_result_payload", side_effect=fake_load_payload):
                summary = run_multi_attempt_for_op(
                    op="elu",
                    category="activation",
                    strategy="one_shot",
                    tool_mode="ascend_search,ascend_fetch",
                    llm_config={"model": "x"},
                    run_dir=run_dir,
                    eval_mode="full",
                    clean_policy="force",
                    max_log_lines=100,
                    max_attempts=3,
                    eval_workers=1,
                    eval_npu=1,
                )

            self.assertEqual(summary["fixed_on_attempt"], 3)
            self.assertEqual(summary["final_status"], "fixed_on_attempt3")
            self.assertIn("attempt3", summary["attempts"])
            self.assertEqual(len(summary["attempts"]), 3)
            self.assertFalse(summary["fixed_in_attempt2"])

    def test_run_multi_attempt_stops_after_attempt1_success(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            p = run_dir / "result_attempt1.json"
            p.write_text("{}", encoding="utf-8")

            def fake_generate(**kwargs):
                attempt_id = kwargs["attempt_id"]
                out_dir = kwargs["out_dir"]
                txt_path = out_dir / f"relu_a{attempt_id}.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text("code", encoding="utf-8")
                return {
                    "result": SimpleNamespace(generated_code=f"code_a{attempt_id}"),
                    "txt_path": txt_path,
                    "report_path": out_dir / f"relu_a{attempt_id}_report.json",
                }

            with patch("generator.scripts.run_agent_multi_rounds._generate_one_attempt", side_effect=fake_generate), \
                 patch("generator.scripts.run_agent_multi_rounds._run_eval_for_txt", return_value=0), \
                 patch("generator.scripts.run_agent_multi_rounds._eval_result_json_path", return_value=p), \
                 patch(
                     "generator.scripts.run_agent_multi_rounds._load_result_payload",
                     return_value={"result": {"relu": {"compiled": True, "correctness": True, "correctness_info": ""}}, "meta": {"logs": {}}},
                 ):
                summary = run_multi_attempt_for_op(
                    op="relu",
                    category="activation",
                    strategy="one_shot",
                    tool_mode="ascend_search,ascend_fetch",
                    llm_config={"model": "x"},
                    run_dir=run_dir,
                    eval_mode="full",
                    clean_policy="force",
                    max_log_lines=100,
                    max_attempts=3,
                    eval_workers=1,
                    eval_npu=1,
                )

            self.assertEqual(summary["fixed_on_attempt"], 1)
            self.assertEqual(summary["final_status"], "pass_on_attempt1")
            self.assertEqual(list(summary["attempts"].keys()), ["attempt1"])


if __name__ == "__main__":
    unittest.main()
