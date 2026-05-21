from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import TestCase


def _load_code_diff_module():
    """Load code_diff without importing generator.repair_memory package __init__ (avoids openai)."""
    path = Path(__file__).resolve().parent.parent / "code_diff.py"
    spec = importlib.util.spec_from_file_location("repair_memory_code_diff_standalone", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCodeDiffForReview(TestCase):
    def test_diff_shows_change(self) -> None:
        m = _load_code_diff_module()
        d = m.format_attempt_code_diff("a = 1\n", "a = 2\n", max_chars=5000)
        self.assertIn("-a = 1", d)
        self.assertIn("+a = 2", d)

    def test_diff_truncation(self) -> None:
        m = _load_code_diff_module()
        prev = "x\n" * 5000
        curr = "y\n" * 5000
        d = m.format_attempt_code_diff(prev, curr, max_chars=200)
        self.assertLessEqual(len(d), 200)
        self.assertIn("truncated", d)


if __name__ == "__main__":
    import unittest

    unittest.main()
