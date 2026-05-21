from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from generator.scripts.multi_round_continue import (
    build_continue_plan,
    find_latest_aggregate_summary,
    merge_op_summary,
)
from generator.scripts.run_agent_multi_rounds import run_multi_attempt_for_op


class TestMultiRoundContinue(unittest.TestCase):
    def _write_aggregate(self, run_dir: Path, max_attempts: int, ops: dict) -> None:
        path = run_dir / f"attempts{max_attempts}_summary_all_ops.json"
        path.write_text(
            json.dumps(
                {
                    "max_attempts": max_attempts,
                    "operator_keys": list(ops.keys()),
                    "ops": ops,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_find_latest_aggregate_summary(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "attempts3_summary_all_ops.json").write_text("{}", encoding="utf-8")
            (run_dir / "attempts5_summary_all_ops.json").write_text("{}", encoding="utf-8")
            p, n = find_latest_aggregate_summary(run_dir, "ascendc")
            self.assertEqual(n, 5)
            self.assertEqual(p.name, "attempts5_summary_all_ops.json")

    def test_build_continue_plan_skip_passed(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "attempt4").mkdir()
            (run_dir / "attempt4" / "relu.txt").write_text("code4", encoding="utf-8")
            self._write_aggregate(
                run_dir,
                5,
                {
                    "relu": {
                        "fixed_on_attempt": 4,
                        "attempts": {
                            "attempt4": {
                                "compiled": True,
                                "correctness": True,
                            }
                        },
                    }
                },
            )
            plan = build_continue_plan(
                run_dir=run_dir,
                new_max_attempts=10,
                kind="ascendc",
            )
            relu = next(e for e in plan.entities if e.entity_key == "relu")
            self.assertEqual(relu.action, "skip_passed")

    def test_build_continue_plan_uses_last_attempt_not_compile_pass(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            for n in (3, 4):
                d = run_dir / f"attempt{n}"
                d.mkdir()
                (d / "gelu.txt").write_text(f"code{n}", encoding="utf-8")
                (d / "gelu_repair_context.txt").write_text(f"log{n}", encoding="utf-8")
            self._write_aggregate(
                run_dir,
                5,
                {
                    "gelu": {
                        "fixed_on_attempt": None,
                        "attempts": {
                            "attempt3": {"compiled": True, "correctness": False},
                            "attempt4": {"compiled": False, "correctness": False},
                        },
                    }
                },
            )
            plan = build_continue_plan(
                run_dir=run_dir,
                new_max_attempts=10,
                kind="ascendc",
            )
            gelu = next(e for e in plan.entities if e.entity_key == "gelu")
            self.assertEqual(gelu.action, "continue")
            self.assertEqual(gelu.last_attempt_id, 4)
            self.assertIn("code4", gelu.seed_txt_path.read_text(encoding="utf-8"))

    def test_build_continue_plan_max_attempts_must_increase(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            self._write_aggregate(run_dir, 5, {"relu": {}})
            with self.assertRaises(ValueError):
                build_continue_plan(run_dir=run_dir, new_max_attempts=5, kind="ascendc")

    def test_merge_op_summary_keeps_prior_attempts(self):
        merged = merge_op_summary(
            {"op": "relu", "attempts": {"attempt1": {"attempt_id": 1}}, "final_status": "old"},
            {"attempts": {"attempt6": {"attempt_id": 6}}, "final_status": "failed_after_attempt10"},
            new_max_attempts=10,
            continued_from_abs="/tmp/run4",
            source_last_attempt=5,
            continue_session_at="2026-05-15T00:00:00Z",
            ran_new_attempts=True,
        )
        self.assertIn("attempt1", merged["attempts"])
        self.assertIn("attempt6", merged["attempts"])
        self.assertEqual(merged["max_attempts"], 10)
        self.assertIn("continue_meta", merged)

    def test_run_multi_attempt_resume_from_attempt3(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "attempt2").mkdir()
            (run_dir / "attempt2" / "elu_repair_context.txt").write_text("prev err", encoding="utf-8")
            result_paths = {3: run_dir / "result_attempt3.json"}
            result_paths[3].write_text("{}", encoding="utf-8")

            def fake_generate(**kwargs):
                attempt_id = kwargs["attempt_id"]
                out_dir = kwargs["out_dir"]
                txt_path = out_dir / f"elu_a{attempt_id}.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(f"code{attempt_id}", encoding="utf-8")
                return {
                    "result": SimpleNamespace(generated_code=f"code{attempt_id}"),
                    "txt_path": txt_path,
                    "report_path": out_dir / f"elu_a{attempt_id}_report.json",
                }

            payloads = [
                {
                    "result": {"elu": {"compiled": True, "correctness": True, "correctness_info": ""}},
                    "meta": {"logs": {}},
                }
            ]

            with patch(
                "generator.scripts.run_agent_multi_rounds._generate_one_attempt",
                side_effect=fake_generate,
            ), patch(
                "generator.scripts.run_agent_multi_rounds._run_eval_for_txt",
                return_value=0,
            ), patch(
                "generator.scripts.run_agent_multi_rounds._eval_result_json_path",
                return_value=result_paths[3],
            ), patch(
                "generator.scripts.run_agent_multi_rounds._load_result_payload",
                return_value=payloads[0],
            ):
                summary = run_multi_attempt_for_op(
                    op="elu",
                    category="activation",
                    strategy="one_shot",
                    tool_mode="no_tool",
                    llm_config={"model": "x"},
                    run_dir=run_dir,
                    eval_mode="full",
                    clean_policy="force",
                    max_log_lines=100,
                    max_attempts=3,
                    eval_workers=1,
                    eval_npu=1,
                    start_attempt_id=3,
                    seed_previous_code="code2",
                    seed_repair_context_path=str(run_dir / "attempt2" / "elu_repair_context.txt"),
                    prior_attempts={
                        "attempt1": {"attempt_id": 1},
                        "attempt2": {"attempt_id": 2},
                    },
                )
            self.assertEqual(summary["fixed_on_attempt"], 3)
            self.assertIn("attempt1", summary["attempts"])
            self.assertIn("attempt2", summary["attempts"])
            self.assertIn("attempt3", summary["attempts"])

    def test_build_continue_plan_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "attempt2").mkdir()
            (run_dir / "attempt2" / "ca6k_00055.txt").write_text("c2", encoding="utf-8")
            (run_dir / "attempt2" / "ca6k_00055_repair_context.txt").write_text("e2", encoding="utf-8")
            agg = run_dir / "attempts2_summary_all_rows.json"
            agg.write_text(
                json.dumps(
                    {
                        "rows": {
                            "ca6k_00055": {
                                "row_index": 55,
                                "fixed_on_attempt": None,
                                "attempts": {
                                    "attempt2": {"compiled": False, "correctness": False},
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            plan = build_continue_plan(
                run_dir=run_dir,
                new_max_attempts=5,
                kind="cuda",
            )
            ent = plan.entities[0]
            self.assertEqual(ent.action, "continue")
            self.assertEqual(ent.row_index, 55)


if __name__ == "__main__":
    unittest.main()
