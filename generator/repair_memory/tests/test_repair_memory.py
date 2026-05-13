from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from generator.repair_memory.failure_stage import infer_failure_stage
from generator.repair_memory.merge import merge_run_inbox
from generator.repair_memory.paths import run_slug_from_run_dir
from generator.repair_memory.schema import SCHEMA_VERSION, validate_record
from generator.repair_memory.tier_gate import classify_tier_and_gates, code_digest
from generator.repair_memory.inject import memory_entries_for_report


class TestMemoryReportEntries(TestCase):
    def test_memory_entries_for_report_shape(self) -> None:
        recs = [
            {
                "memory_id": "u1",
                "tier": "A",
                "confidence": "high",
                "op_key": "gelu",
                "category": "activation",
                "tool_mode": "dma_alignment_engine",
                "eval_mode": "full",
                "transition": {"compiled": [False, True]},
                "failure_stage_before": "02-build",
                "failure_stage_after": "correctness",
                "error_anchors_before": "e0",
                "error_anchors_after": "e1",
                "natural_language": "当编译失败时不要乱改，应检查 OPP。",
            }
        ]
        rows = memory_entries_for_report(recs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["memory_id"], "u1")
        self.assertEqual(rows[0]["display_order"], 1)
        self.assertIn("natural_language", rows[0])


class TestTierGate(TestCase):
    def test_tier_a_compile_fixed(self) -> None:
        op = "gelu"
        prev_out = {"eval_ran": True, "compiled": False, "correctness": None}
        curr_out = {"eval_ran": True, "compiled": True, "correctness": False}
        prev_pl = {"result": {op: {"compiled": False, "correctness": None}}, "meta": {"logs": {}}}
        curr_pl = {"result": {op: {"compiled": True, "correctness": False}}, "meta": {"logs": {}}}
        tier, _ = classify_tier_and_gates(
            op=op,
            prev_outcome=prev_out,
            curr_outcome=curr_out,
            prev_payload=prev_pl,
            curr_payload=curr_pl,
            prev_code="a",
            curr_code="b",
        )
        self.assertEqual(tier, "A")

    def test_no_write_same_code(self) -> None:
        op = "gelu"
        prev_out = curr_out = {"eval_ran": True, "compiled": False, "correctness": None}
        pl = {"result": {op: {"compiled": False, "correctness": None}}, "meta": {"logs": {}}}
        tier, reason = classify_tier_and_gates(
            op=op,
            prev_outcome=prev_out,
            curr_outcome=curr_out,
            prev_payload=pl,
            curr_payload=pl,
            prev_code="same",
            curr_code="same",
        )
        self.assertIsNone(tier)
        self.assertEqual(reason, "no_code_change")


class TestMergeInbox(TestCase):
    def test_merge_moves_into_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            slug = "test_run"
            inbox = root / "inbox" / slug
            inbox.mkdir(parents=True)
            rec = {
                "memory_id": "mid-1",
                "schema_version": SCHEMA_VERSION,
                "tier": "A",
                "confidence": "high",
                "op_key": "x",
                "category": "c",
                "tool_mode": "t",
                "strategy": "s",
                "eval_mode": "full",
                "transition": {"compiled": [False, True]},
                "failure_stage_before": "02-build",
                "failure_stage_after": "correctness",
                "error_anchors_before": "e1",
                "error_anchors_after": "e2",
                "code_digest_before": "1",
                "code_digest_after": "2",
                "natural_language": "当编译失败时不要乱改，应检查 CMake 与 OPP 路径。",
                "evidence_refs": [],
            }
            self.assertTrue(validate_record(rec))
            fp = inbox / "mem_testuuid.jsonl"
            fp.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            n = merge_run_inbox(root, slug)
            self.assertEqual(n, 1)
            canon = root / "canonical" / "repair_memories.jsonl"
            self.assertTrue(canon.is_file())
            lines = canon.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertFalse(fp.exists())


class TestFailureStage(TestCase):
    def test_infer_correctness_stage(self) -> None:
        s = infer_failure_stage(
            compiled=True,
            correctness=False,
            meta_logs={"06-eval": "x"},
        )
        self.assertEqual(s, "correctness")


class TestRunSlug(TestCase):
    def test_slug_no_slash(self) -> None:
        p = Path("/a/b/c/d")
        s = run_slug_from_run_dir(p)
        self.assertNotIn("/", s)
