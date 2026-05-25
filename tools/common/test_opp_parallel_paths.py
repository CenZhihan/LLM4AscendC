#!/usr/bin/env python3
"""Regression: parallel OPP paths must not nest _parallel_w* segments."""
from __future__ import annotations

import os
import unittest

from tools.common.env import (
    _ASCEND_CUSTOM_OPP_BASE_ENV,
    _ASCEND_CUSTOM_OPP_ENV,
    apply_agent_parallel_slot_env,
    init_parallel_op_slot_os_environ,
    parallel_opp_path_for_bucket,
    resolve_ascend_custom_opp_base,
)


class TestOppParallelPaths(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved)

    def test_resolve_strips_nested_parallel_dirs(self) -> None:
        nested = "/tmp/ascend_custom_opp/_parallel_w1/_parallel_w2/_parallel_w0"
        self.assertEqual(
            resolve_ascend_custom_opp_base(nested),
            "/tmp/ascend_custom_opp",
        )

    def test_apply_agent_slot_does_not_nest(self) -> None:
        base = "/tmp/ascend_custom_opp_test"
        os.environ[_ASCEND_CUSTOM_OPP_ENV] = base
        apply_agent_parallel_slot_env(op_slot=2, parallel_ops=4, npu_count=4)
        first = os.environ[_ASCEND_CUSTOM_OPP_ENV]
        self.assertEqual(first, f"{base}/_parallel_w2")

        apply_agent_parallel_slot_env(op_slot=8, parallel_ops=4, npu_count=4)
        second = os.environ[_ASCEND_CUSTOM_OPP_ENV]
        self.assertEqual(second, f"{base}/_parallel_w0")
        self.assertNotIn("_parallel_w2/_parallel", second)

    def test_init_parallel_op_slot_idempotent(self) -> None:
        base = "/tmp/ascend_custom_opp_test2"
        os.environ[_ASCEND_CUSTOM_OPP_ENV] = f"{base}/_parallel_w3"
        init_parallel_op_slot_os_environ(op_slot=1, parallel_ops=4, npu_count=4, label="t")
        self.assertEqual(os.environ[_ASCEND_CUSTOM_OPP_ENV], f"{base}/_parallel_w1")
        self.assertEqual(os.environ[_ASCEND_CUSTOM_OPP_BASE_ENV], base)

    def test_parallel_opp_path_for_bucket(self) -> None:
        p = parallel_opp_path_for_bucket(
            base_opp="/data/opp/_parallel_w9",
            bucket=2,
        )
        self.assertEqual(p, "/data/opp/_parallel_w2")


if __name__ == "__main__":
    unittest.main()
