from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generator.scripts.run_agent_multi_rounds import _synthetic_result_payload_when_eval_json_missing


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestEvalOperatorMaterializeFailureJson(unittest.TestCase):
    def test_materialize_failure_writes_result_json_under_artifacts(self):
        """
        Invalid txt (missing MKB blocks) fails before _execute_pipeline; eval_operator must
        still emit result_<stem>.json so multi-attempt runners can load correctness_info.
        """
        with tempfile.TemporaryDirectory() as td:
            txt = Path(td) / "bad_op.txt"
            txt.write_text("# not a bundle — missing triple-quoted vars\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "eval_operator.py"),
                "--txt",
                str(txt),
                "--mode",
                "full",
                "--clean-policy",
                "force",
            ]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            self.assertNotEqual(proc.returncode, 0)
            result_json = REPO_ROOT / "artifacts" / "bad_op" / "result_bad_op.json"
            self.assertTrue(result_json.is_file(), msg=f"expected {result_json}")
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertIn("result", payload)
            self.assertIn("bad_op", payload["result"])
            self.assertEqual(payload["result"]["bad_op"]["compiled"], False)
            self.assertEqual(payload["result"]["bad_op"]["correctness"], False)
            info = payload["result"]["bad_op"]["correctness_info"]
            self.assertTrue(
                "parse_txt_bundle" in info or "missing blocks" in info or "ValueError" in info,
                msg=info[:500],
            )


class TestSyntheticEvalMissingJsonPayload(unittest.TestCase):
    def test_shape_matches_eval_operator_schema(self):
        p = Path("/tmp/ghost.json")
        pl = _synthetic_result_payload_when_eval_json_missing(
            op="relu",
            eval_mode="full",
            eval_rc=1,
            result_path=p,
        )
        self.assertEqual(pl["result"]["relu"]["compiled"], False)
        self.assertEqual(pl["result"]["relu"]["correctness"], False)
        self.assertIn("did not emit result json", pl["result"]["relu"]["correctness_info"])
        self.assertIn(str(p), pl["result"]["relu"]["correctness_info"])
        self.assertEqual(pl["meta"]["mode"], "full")
        self.assertEqual(pl["meta"]["logs"], {})
