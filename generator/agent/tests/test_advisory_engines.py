"""Tests for dtype_policy_engine and dma_alignment_engine advisory logic."""
from __future__ import annotations

import json
import unittest

from generator.agent.agent_config import normalize_tool_choice_name
from generator.agent.reporting.advisory_report import ADVISORY_SEPARATOR, advisory_display_string
from generator.agent.rules.dtype_policy_engine import analyze_dtype_policy
from generator.agent.rules.dma_alignment_engine import analyze_dma_alignment


class TestToolNameNormalization(unittest.TestCase):
    def test_new_builtins_alias(self) -> None:
        self.assertEqual(normalize_tool_choice_name("DTYPE_POLICY_ENGINE"), "dtype_policy_engine")
        self.assertEqual(normalize_tool_choice_name("dma_alignment_engine"), "dma_alignment_engine")


class TestDtypePolicyEngine(unittest.TestCase):
    def test_invalid_mode_fallback(self) -> None:
        r = analyze_dtype_policy("", {"target_precision_mode": "not_a_mode"})
        self.assertIn("unknown target_precision_mode", r["parse_warnings"][0])
        self.assertEqual(r["target_precision_mode"], "match_pytorch")

    def test_matmul_fp16_accum_fp32(self) -> None:
        r = analyze_dtype_policy(
            "",
            {
                "op_family": "matmul_like",
                "target_precision_mode": "match_pytorch",
                "io_dtypes": {"io": "float16"},
            },
        )
        acc = [x for x in r["stages"] if x["stage"] == "accumulate"][0]
        self.assertEqual(acc["recommended_compute_dtype"], "float32")

    def test_fp32_accum_fp16_io_explicit(self) -> None:
        r = analyze_dtype_policy("", {"target_precision_mode": "fp32_accum_fp16_io"})
        self.assertTrue(any("FP32 accumulation" in x["note"] for x in r["stages"]))


class TestDmaAlignmentEngine(unittest.TestCase):
    def test_gm_to_ub_misaligned_suggests_pad(self) -> None:
        r = analyze_dma_alignment(
            "",
            {
                "transfers": [
                    {"direction": "gm_to_ub", "dtype": "half", "elem_count": 7, "gm_offset_bytes": 0}
                ]
            },
        )
        self.assertTrue(any(i["code"] == "prefer_datacopy_pad_gm_ub" for i in r["issues"]))
        self.assertEqual(r["transfers"][0]["suggested_api"], "DataCopyPad")

    def test_compare_256b_violation(self) -> None:
        r = analyze_dma_alignment(
            "",
            {
                "transfers": [
                    {
                        "direction": "ub_to_ub",
                        "dtype": "float32",
                        "elem_count": 15,
                        "involves_api": "Compare",
                    }
                ]
            },
        )
        self.assertTrue(any(i["code"] == "compare_not_256b" for i in r["issues"]))

    def test_advisory_display_has_separator_and_json_prefix(self) -> None:
        r = analyze_dtype_policy("q", {"op_family": "elementwise"})
        s = advisory_display_string(r)
        self.assertIn(ADVISORY_SEPARATOR, s)
        prefix = s.split(ADVISORY_SEPARATOR, 1)[0].strip()
        obj = json.loads(prefix)
        self.assertEqual(obj["tool"], "dtype_policy_engine")


if __name__ == "__main__":
    unittest.main()
