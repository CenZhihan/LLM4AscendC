"""Tests for generator.agent.memory backends."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import generator.agent.memory.tencentdb
from generator.agent.memory import (
    create_memory_backend,
    NullMemoryBackend,
    LocalRepairMemoryBackend,
    TencentDBMemoryBackend,
    RecallResult,
)


class TestNullMemoryBackend(unittest.TestCase):
    def test_recall_returns_empty(self):
        backend = NullMemoryBackend()
        result = backend.recall(query="test", session_key="s1")
        self.assertEqual(result.text, "")
        self.assertEqual(result.backend, "null")

    def test_write_noop(self):
        backend = NullMemoryBackend()
        backend.write(session_key="s1", user_content="u", assistant_content="a")
        # no exception raised

    def test_close_noop(self):
        backend = NullMemoryBackend()
        backend.close()
        # no exception raised


class TestLocalRepairMemoryBackend(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmpdir.name)
        self.backend = LocalRepairMemoryBackend(run_dir=self.run_dir, op="gelu")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_write_creates_repair_context(self):
        self.backend.write(
            session_key="s1",
            user_content="repair text",
            assistant_content="code",
            metadata={"attempt_id": 1, "repair_text": "repair text"},
        )
        path = self.run_dir / "attempt1" / "gelu_repair_context.txt"
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(encoding="utf-8"), "repair text")

    def test_recall_reads_previous_attempt(self):
        # Write attempt 1
        self.backend.write(
            session_key="s1",
            user_content="error context",
            assistant_content="code",
            metadata={"attempt_id": 1, "repair_text": "error context"},
        )
        # Recall for attempt 2
        result = self.backend.recall(
            query="q",
            session_key="s1",
            metadata={"attempt_id": 2},
        )
        self.assertEqual(result.text, "error context")
        self.assertEqual(result.backend, "local")

    def test_recall_returns_empty_when_no_previous(self):
        result = self.backend.recall(
            query="q",
            session_key="s1",
            metadata={"attempt_id": 2},
        )
        self.assertEqual(result.text, "")

    def test_recall_uses_previous_repair_context_path_hint(self):
        hint_path = self.run_dir / "attempt1" / "gelu_repair_context.txt"
        hint_path.parent.mkdir(parents=True, exist_ok=True)
        hint_path.write_text("hinted text", encoding="utf-8")
        result = self.backend.recall(
            query="q",
            session_key="s1",
            metadata={"attempt_id": 2, "previous_repair_context_path": str(hint_path)},
        )
        self.assertEqual(result.text, "hinted text")


class TestTencentDBMemoryBackend(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmpdir.name)
        # Inject a mock requests module so patching works even when real requests is absent
        self._orig_requests = generator.agent.memory.tencentdb.requests
        self.mock_requests = MagicMock()
        generator.agent.memory.tencentdb.requests = self.mock_requests

    def tearDown(self):
        generator.agent.memory.tencentdb.requests = self._orig_requests
        self.tmpdir.cleanup()

    def test_health_check_ok(self):
        self.mock_requests.get.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "ok"}
        )
        backend = TencentDBMemoryBackend(url="http://localhost:8420")
        self.assertTrue(backend.health_check())

    def test_health_check_degraded(self):
        self.mock_requests.get.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "degraded"}
        )
        backend = TencentDBMemoryBackend(url="http://localhost:8420")
        self.assertTrue(backend.health_check())

    def test_health_check_fail(self):
        self.mock_requests.get.side_effect = Exception("connection refused")
        backend = TencentDBMemoryBackend(url="http://localhost:8420")
        self.assertFalse(backend.health_check())

    def test_recall_returns_context(self):
        self.mock_requests.post.return_value = MagicMock(
            status_code=200, json=lambda: {"context": "recalled memory"}
        )
        backend = TencentDBMemoryBackend(
            url="http://localhost:8420",
            keep_local_repair_context=False,
        )
        result = backend.recall(query="op=gelu", session_key="s1")
        self.assertIn("recalled memory", result.text)
        self.assertEqual(result.backend, "tencentdb")
        self.mock_requests.post.assert_called_once()
        _, kwargs = self.mock_requests.post.call_args
        self.assertEqual(kwargs["json"]["query"], "op=gelu")
        self.assertEqual(kwargs["json"]["session_key"], "s1")

    def test_write_calls_capture(self):
        self.mock_requests.post.return_value = MagicMock(
            status_code=200, json=lambda: {"l0_recorded": 1}
        )
        backend = TencentDBMemoryBackend(
            url="http://localhost:8420",
            keep_local_repair_context=False,
        )
        backend.write(
            session_key="s1",
            user_content="repair text",
            assistant_content="generated code",
        )
        self.mock_requests.post.assert_called_once()
        _, kwargs = self.mock_requests.post.call_args
        self.assertEqual(kwargs["json"]["user_content"], "repair text")
        self.assertEqual(kwargs["json"]["assistant_content"], "generated code")
        self.assertEqual(kwargs["json"]["session_key"], "s1")

    def test_recall_merges_local_and_tencent(self):
        self.mock_requests.post.return_value = MagicMock(
            status_code=200, json=lambda: {"context": "long term"}
        )
        backend = TencentDBMemoryBackend(
            url="http://localhost:8420",
            keep_local_repair_context=True,
            run_dir=self.run_dir,
            op="gelu",
        )
        # Seed local repair context for attempt 1
        backend.local.write(
            session_key="s1",
            user_content="local error",
            assistant_content="code",
            metadata={"attempt_id": 1, "repair_text": "local error"},
        )
        result = backend.recall(
            query="q",
            session_key="s1",
            metadata={"attempt_id": 2},
        )
        self.assertIn("local error", result.text)
        self.assertIn("long term", result.text)
        self.assertIn("Long-term memory recall:", result.text)


class TestFactory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_create_off(self):
        backend = create_memory_backend("off", run_dir=self.run_dir, op="gelu")
        self.assertIsInstance(backend, NullMemoryBackend)

    def test_create_local(self):
        backend = create_memory_backend("local", run_dir=self.run_dir, op="gelu")
        self.assertIsInstance(backend, LocalRepairMemoryBackend)

    @patch("generator.agent.memory.factory.TencentDBMemoryBackend")
    def test_create_tencentdb(self, mock_cls):
        mock_inst = MagicMock()
        mock_inst.health_check.return_value = True
        mock_cls.return_value = mock_inst
        backend = create_memory_backend("tencentdb", run_dir=self.run_dir, op="gelu")
        self.assertEqual(backend, mock_inst)

    @patch("generator.agent.memory.factory.TencentDBMemoryBackend")
    def test_create_tencentdb_fallback_on_health_failure(self, mock_cls):
        mock_inst = MagicMock()
        mock_inst.health_check.return_value = False
        mock_cls.return_value = mock_inst
        backend = create_memory_backend("tencentdb", run_dir=self.run_dir, op="gelu")
        self.assertIsInstance(backend, LocalRepairMemoryBackend)

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_memory_backend("unknown", run_dir=self.run_dir, op="gelu")


if __name__ == "__main__":
    unittest.main()
