from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from generator.repair_memory.failure_stage import infer_failure_stage
from generator.repair_memory.llm_util import assistant_message_text
from generator.repair_memory.merge import merge_run_inbox
from generator.repair_memory.paths import run_slug_from_run_dir
from generator.repair_memory.schema import SCHEMA_VERSION, validate_record
from generator.repair_memory.tier_gate import classify_tier_and_gates, code_digest
from generator.repair_memory.inject import memory_entries_for_report
from generator.repair_memory.select import parse_memory_selection_output


class TestAssistantMessageText(TestCase):
    def test_prefers_content_over_reasoning(self) -> None:
        class Msg:
            content = "  hello  "
            reasoning_content = "reason"

        self.assertEqual(assistant_message_text(Msg()), "hello")

    def test_fallback_reasoning_content(self) -> None:
        class Msg:
            content = ""
            reasoning_content = '{"memory_ids": [], "selection_rationale": "x"}'

        self.assertTrue(assistant_message_text(Msg()).startswith("{"))


class TestParseMemorySelectionOutput(TestCase):
    def test_parses_ids_and_rationale(self) -> None:
        raw = (
            '{"memory_ids": ["u1", "u2"], "selection_rationale": '
            '"u1 matches txt bundle; u2 matches CPack stage."}'
        )
        d = parse_memory_selection_output(raw, max_n=5)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["memory_ids"], ["u1", "u2"])
        self.assertEqual(d["selection_rationale"], "u1 matches txt bundle; u2 matches CPack stage.")
        self.assertEqual(d["parse_error"], "")

    def test_empty_ids_with_rationale(self) -> None:
        raw = '{"memory_ids": [], "selection_rationale": "No manifest line matches this log."}'
        d = parse_memory_selection_output(raw, max_n=5)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["memory_ids"], [])
        self.assertEqual(d["selection_rationale"], "No manifest line matches this log.")

    def test_omitted_rationale_placeholder(self) -> None:
        d = parse_memory_selection_output('{"memory_ids": ["x"]}', max_n=5)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["memory_ids"], ["x"])
        self.assertEqual(d["selection_rationale"], "(model omitted selection_rationale)")

    def test_respects_max_n(self) -> None:
        raw = '{"memory_ids": ["a","b","c"], "selection_rationale": "x"}'
        d = parse_memory_selection_output(raw, max_n=2)
        self.assertEqual(d["memory_ids"], ["a", "b"])

    def test_no_json_object(self) -> None:
        d = parse_memory_selection_output("not json", max_n=5)
        self.assertFalse(d["parse_ok"])
        self.assertEqual(d["memory_ids"], [])
        self.assertEqual(d["parse_error"], "no_json_object_in_output")

    def test_picks_first_json_when_reply_leads_with_json(self) -> None:
        raw = '{"memory_ids": ["only"], "selection_rationale": "first"}\nextra note not json'
        d = parse_memory_selection_output(raw, max_n=5)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["memory_ids"], ["only"])
        self.assertEqual(d["selection_rationale"], "first")

    def test_picks_last_json_when_multiple_memory_ids_objects(self) -> None:
        raw = (
            'CoT echoes example {"memory_ids": ["fake"], "selection_rationale": "x"}\n'
            'Answer {"memory_ids": [], "selection_rationale": "No line matches CPack error."}'
        )
        d = parse_memory_selection_output(raw, max_n=5)
        self.assertTrue(d["parse_ok"])
        self.assertEqual(d["memory_ids"], [])
        self.assertEqual(d["selection_rationale"], "No line matches CPack error.")


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
                "natural_language": "When compile fails with OPP install errors, do not randomize tiling; instead verify custom OPP path and reinstall.",
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
                "natural_language": "When CPack fails on missing binary config, do not rerun package blindly; instead regenerate CMake outputs and fix paths.",
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
