from __future__ import annotations

import unittest

from tools.common.error_extract import (
    anchor_from_excerpt,
    build_layered_errors_from_log_text,
    extract_root_cause,
    extract_symptom,
    format_layered_correctness_info,
    parse_layered_correctness_info,
)


MASKED_CUMSUM_TAIL = """
[ERROR] kernel compile
/root/proj/op_kernel/masked_cumsum_custom.cpp:79:22: error: no member named 'GetValue' in namespace 'AscendC'
/root/proj/op_kernel/masked_cumsum_custom.cpp:83:22: error: no member named 'SetValue' in namespace 'AscendC'
RuntimeError: Kernel Compilation Error: OpType MaskedCumsumCustom
gmake: *** [Makefile:293: binary] Error 2
[100%] Built target cust_op_proto
CPack: Install projects
CMake Error at cmake_install.cmake:62 (file):
  file INSTALL cannot find ".../binary/config": No such file or directory.
CPack Error: Error when generating package: opp
gmake: *** [Makefile:71: package] Error 1
"""


class TestErrorExtract(unittest.TestCase):
    def test_root_cause_before_cpack(self) -> None:
        root = extract_root_cause(MASKED_CUMSUM_TAIL)
        self.assertIn("GetValue", root)
        self.assertNotIn("binary/config", root)

    def test_symptom_is_cpack(self) -> None:
        sym = extract_symptom(MASKED_CUMSUM_TAIL)
        self.assertTrue("CPack" in sym or "binary/config" in sym or "INSTALL" in sym)
        self.assertNotIn("GetValue", sym)

    def test_cpack_only_log(self) -> None:
        log = """
CMake Error at cmake_install.cmake:62 (file):
  file INSTALL cannot find ".../binary/config"
CPack Error: Error when generating package
"""
        root = extract_root_cause(log)
        self.assertEqual(root, "")
        sym = extract_symptom(log)
        self.assertIn("binary/config", sym)

    def test_layered_format_roundtrip(self) -> None:
        root, sym, formatted = build_layered_errors_from_log_text(MASKED_CUMSUM_TAIL)
        self.assertIn("GetValue", root)
        self.assertIn("=== root_cause ===", formatted)
        self.assertIn("=== symptom ===", formatted)
        r2, s2 = parse_layered_correctness_info(formatted)
        self.assertIn("GetValue", r2)
        self.assertTrue("CPack" in s2 or "INSTALL" in s2)

    def test_anchor_from_root(self) -> None:
        root, _, _ = build_layered_errors_from_log_text(MASKED_CUMSUM_TAIL)
        a = anchor_from_excerpt(root)
        self.assertIn("GetValue", a)


if __name__ == "__main__":
    unittest.main()
