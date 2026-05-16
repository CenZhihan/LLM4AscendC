from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generator.repair_memory.error_signals import (
    build_attempt_error_bundle,
    format_repair_error_context,
    select_error_log_paths,
)
from generator.repair_memory.review_llm import _has_template_parts
from tools.common.tests.test_error_extract import MASKED_CUMSUM_TAIL


class TestErrorSignals(unittest.TestCase):
    def test_select_log_paths_prefers_build(self) -> None:
        logs = {"01-msopgen": "/a.log", "02-build": "/b.log", "06-eval": "/c.log"}
        paths = select_error_log_paths(logs)
        self.assertEqual(paths[0], "/b.log")
        self.assertIn("/c.log", paths)

    def test_bundle_from_payload_with_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "build.log"
            log_path.write_text(MASKED_CUMSUM_TAIL, encoding="utf-8")
            payload = {
                "result": {
                    "masked_cumsum": {
                        "compiled": False,
                        "correctness": False,
                        "correctness_info": "=== symptom ===\nCPack Error\n",
                    }
                },
                "meta": {"logs": {"02-build": str(log_path)}},
            }
            bundle = build_attempt_error_bundle(payload, "masked_cumsum", max_log_lines=220)
            self.assertIn("GetValue", bundle.root_cause)
            self.assertIn("GetValue", bundle.root_cause_anchor)
            self.assertTrue(bundle.log_excerpt)
            ctx = format_repair_error_context(op="masked_cumsum", bundle=bundle)
            self.assertIn("root_cause", ctx)
            self.assertIn("log file:", ctx)

    def test_review_rejects_cot_pollution(self) -> None:
        cot = (
            "We need to write a single conditional sentence following the pattern. "
            "When CPack fails, do not ignore it; instead fix binary/config."
        )
        self.assertFalse(_has_template_parts(cot))

    def test_review_accepts_valid_sentence(self) -> None:
        good = (
            "When AscendC kernel compile reports no member named 'GetValue', "
            "do not call AscendC::GetValue on LocalTensor; instead use GetValue/SetValue methods."
        )
        self.assertTrue(_has_template_parts(good))


if __name__ == "__main__":
    unittest.main()
