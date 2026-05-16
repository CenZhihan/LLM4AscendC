from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from generator.repair_memory.maintenance import (
    append_removed_archive,
    apply_from_report,
    classify_bucket,
    load_canonical_records,
    phase1_purge,
    purge_reason,
    run_maintenance,
)
from generator.repair_memory.schema import SCHEMA_VERSION


def _base_record(**overrides: object) -> dict:
    rec = {
        "memory_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "schema_version": SCHEMA_VERSION,
        "tier": "A",
        "confidence": "high",
        "op_key": "test_op",
        "category": "activation",
        "tool_mode": "no_tool",
        "strategy": "one_shot",
        "eval_mode": "full",
        "transition": {"compiled": [False, True], "correctness": [False, True]},
        "failure_stage_before": "unknown",
        "failure_stage_after": "success",
        "error_anchors_before": "ValueError: txt bundle missing blocks",
        "error_anchors_after": "",
        "symptom_anchor_before": "ValueError: txt bundle missing blocks",
        "symptom_anchor_after": "",
        "root_cause_anchor_before": "",
        "root_cause_anchor_after": "",
        "code_digest_before": "a",
        "code_digest_after": "b",
        "natural_language": (
            "When parse_txt_bundle raises ValueError for missing host_operator_src, "
            "do not omit required blocks; instead provide host_operator_src, kernel_src, and python_bind_src."
        ),
        "evidence_refs": [],
        "created_at": "2026-05-15T00:00:00+00:00",
    }
    rec.update(overrides)  # type: ignore[arg-type]
    return rec


class TestPurgeRules(TestCase):
    def test_good_record_not_purged(self) -> None:
        self.assertIsNone(purge_reason(_base_record()))

    def test_cot_nl_purged(self) -> None:
        rec = _base_record(
            natural_language="We need to fix the kernel by adding blocks instead of skipping them.",
        )
        self.assertEqual(purge_reason(rec), "invalid_nl_template")

    def test_low_detail_cpack_only_purged(self) -> None:
        rec = _base_record(
            natural_language=(
                "When the build fails during packaging, do not ignore the failure; "
                "instead rerun the build until it succeeds without checking paths."
            ),
            error_anchors_after="CMake Error: file INSTALL cannot find binary/config",
            symptom_anchor_after="CMake Error: file INSTALL cannot find binary/config",
        )
        self.assertEqual(purge_reason(rec), "low_detail")

    def test_phase1_purge_counts(self) -> None:
        good = _base_record(memory_id="good-id")
        bad = _base_record(
            memory_id="bad-id",
            natural_language="Following the pattern we should fix things instead of not fixing.",
        )
        kept, removed = phase1_purge([good, bad])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].reason, "invalid_nl_template")


class TestBucket(TestCase):
    def test_classify_txt_missing(self) -> None:
        rec = _base_record(error_anchors_before="missing blocks: host_operator_src")
        self.assertEqual(classify_bucket(rec), "txt_missing_blocks")


class TestDedupMock(TestCase):
    def test_llm_dedup_drops_one(self) -> None:
        r1 = _base_record(memory_id="11111111-1111-1111-1111-111111111111")
        r2 = _base_record(memory_id="22222222-2222-2222-2222-222222222222")
        r2["natural_language"] = r1["natural_language"]

        def fake_llm(**kwargs: object) -> str:
            return json.dumps(
                {
                    "drop_ids": ["22222222-2222-2222-2222-222222222222"],
                    "rationale": "Duplicate txt missing-blocks advice.",
                }
            )

        result = run_maintenance([r1, r2], skip_dedup=False, llm_call=fake_llm)
        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.kept[0]["memory_id"], "11111111-1111-1111-1111-111111111111")
        self.assertEqual(result.dedup_count, 1)


class TestApplyFromReport(TestCase):
    def test_remove_list_keeps_new_records(self) -> None:
        old = _base_record(memory_id="old-bad")
        new = _base_record(memory_id="new-good", op_key="new_op")
        report = {
            "kept_ids": ["new-good"],
            "removed": [{"memory_id": "old-bad", "phase": "purge", "reason": "test", "bucket": ""}],
        }
        kept, removed, warnings = apply_from_report([old, new], report, mode="remove_list")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["memory_id"], "new-good")
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].record["memory_id"], "old-bad")

    def test_kept_ids_mode(self) -> None:
        a = _base_record(memory_id="aaa")
        b = _base_record(memory_id="bbb")
        report = {"kept_ids": ["aaa"], "removed": []}
        kept, removed, _ = apply_from_report([a, b], report, mode="kept_ids")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["memory_id"], "aaa")
        self.assertEqual(len(removed), 1)


class TestArchiveAndLoad(TestCase):
    def test_load_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "repair_memories.jsonl"
            rec = _base_record()
            p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            loaded = load_canonical_records(p)
            self.assertEqual(len(loaded), 1)
            arch = Path(td) / "removed.jsonl"
            from generator.repair_memory.maintenance import RemovedEntry

            append_removed_archive(
                arch,
                [RemovedEntry(record=rec, phase="purge", reason="test")],
            )
            lines = arch.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["phase"], "purge")
            self.assertIn("record", obj)


if __name__ == "__main__":
    unittest.main()
