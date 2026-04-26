"""Unit tests: tool registry, parse_tool_mode with plugins, JSON tool choice."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from generator.agent.retrievers import env_checker as env_checker_mod
from generator.agent.agent_config import (
    parse_tool_mode,
    tool_mode_to_string,
    normalize_tool_choice_name,
    has_plugin,
)
from generator.agent.nodes.ascend_fetch import ascend_fetch_node
from generator.agent.nodes.ascend_search import ascend_search_node
from generator.agent.query_utils import extract_api_name, extract_chip_name, extract_npu_query_params
from generator.agent.retrievers.api_doc_retriever import ApiDocRetriever, ApiSignatureResult
from generator.agent.retrievers.code_search_snippet_retriever import CodeSearchSnippetRetriever, _build_query_intent
from generator.agent.retrievers.env_checker import ApiCheckResult, NpuDeviceResult
from generator.agent.tool_choice import parse_tool_choice_json
from generator.agent.tool_registry import RegisteredToolSpec, get_tool_registry, register_tool


def _echo_handler(state: dict) -> dict:
    q = (state.get("current_query") or "").strip()
    r = state.get("query_round_count", 0) + 1
    msg = f"[echo] {q}"
    return {
        "registered_tool_results": [msg],
        "query_round_count": r,
        "tool_calls_log": [{"round": r, "tool": "echo_test", "query": q, "response": msg}],
    }


class TestToolChoiceJson(unittest.TestCase):
    def test_parse_valid(self):
        raw = '{"tool":"kb","query":"hello","args":null}'
        c, err = parse_tool_choice_json(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertEqual(c.tool, "kb")
        self.assertEqual(c.query, "hello")

    def test_strip_fence(self):
        raw = '```json\n{"tool":"ANSWER","query":"","args":null}\n```'
        c, err = parse_tool_choice_json(raw)
        self.assertIsNone(err)
        assert c is not None
        self.assertEqual(c.tool, "ANSWER")

    def test_parse_valid_with_thinking(self):
        raw = (
            '{"tool":"kb","query":"DataCopy","args":null,'
            '"thinking":{"goal":"find signature","missing_info":"exact params",'
            '"why_tool":"kb has API docs","expected_output":"signature snippet"}}'
        )
        c, err = parse_tool_choice_json(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertEqual(c.thinking, {
            "goal": "find signature",
            "missing_info": "exact params",
            "why_tool": "kb has API docs",
            "expected_output": "signature snippet",
        })

    def test_parse_valid_with_trailing_extra_brace(self):
        raw = '{"tool":"api_lookup","query":"","args":{"api_name":"AscendC::Muls"}}}'
        c, err = parse_tool_choice_json(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertEqual(c.tool, "api_lookup")
        self.assertEqual(c.args, {"api_name": "AscendC::Muls"})

    def test_parse_skips_invalid_json_object_before_valid_one(self):
        raw = (
            '{"goal":"find docs"}\n'
            '{"tool":"api_constraint","query":"","args":{"api_name":"AscendC::Matmul"}}'
        )
        c, err = parse_tool_choice_json(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertEqual(c.tool, "api_constraint")
        self.assertEqual(c.args, {"api_name": "AscendC::Matmul"})


class TestQueryUtils(unittest.TestCase):
    def test_extract_api_name_ignores_meta_words(self):
        self.assertEqual(
            extract_api_name(
                "Need the exact signature details for AscendC::DataCopy, not generic signatures",
                known_names=["DataCopy", "Muls"],
            ),
            "AscendC::DataCopy",
        )

    def test_extract_api_name_prefers_args(self):
        self.assertEqual(
            extract_api_name("API: signatures", args={"api_name": "MatmulType"}),
            "MatmulType",
        )

    def test_extract_npu_query_params(self):
        query_type, device_id = extract_npu_query_params(
            "check NPU memory on device 1",
            args={"query_type": "memory", "device_id": 1},
        )
        self.assertEqual(query_type, "memory")
        self.assertEqual(device_id, 1)

    def test_extract_chip_name_prefers_chip_token_over_meta_word(self):
        self.assertEqual(
            extract_chip_name(
                "Ascend910B2 chip specs",
                known_names=["Ascend910B2", "Ascend910B"],
            ),
            "Ascend910B2",
        )

    def test_extract_chip_name_prefers_args(self):
        self.assertEqual(
            extract_chip_name(
                "chip specs",
                args={"chip_name": "Ascend950DT"},
                known_names=["Ascend950DT"],
            ),
            "Ascend950DT",
        )


class TestApiRetrieverFallback(unittest.TestCase):
    def test_lookup_signature_uses_local_headers_and_doc_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            include_root = os.path.join(tmp, "include", "basic_api")
            os.makedirs(include_root, exist_ok=True)
            with open(os.path.join(include_root, "matmul.h"), "w", encoding="utf-8") as handle:
                handle.write("using MatmulType = int;\n")
            with open(os.path.join(tmp, "api-matmul.md"), "w", encoding="utf-8") as handle:
                handle.write("# Matmul Guide\n\n## 类型\nMatmulType is used for matmul kernels.\n")

            retriever = ApiDocRetriever(knowledge_path=tmp)
            result = retriever.lookup_signature("MatmulType")

        self.assertEqual(result.api_name, "MatmulType")
        self.assertIn("MatmulType", result.signature)
        self.assertIn("matmul.h", result.header_files)
        self.assertTrue(result.doc_metadata)
        self.assertEqual(result.doc_metadata[0]["path"], "api-matmul.md")

    def test_lookup_signature_ignores_non_signature_header_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            include_root = os.path.join(tmp, "include", "basic_api")
            os.makedirs(include_root, exist_ok=True)
            with open(os.path.join(include_root, "fusion_gelu.h"), "w", encoding="utf-8") as handle:
                handle.write("enum class GeluApproxiMate : uint8_t { ERF = 0, TANH = 1 };\n")

            retriever = ApiDocRetriever(knowledge_path=tmp)
            result = retriever.lookup_signature("Tanh")

        self.assertEqual(result.signature, "")
        self.assertIn("未找到 API 'Tanh'", result.details)


class TestApiAndEnvNodes(unittest.TestCase):
    def test_api_lookup_node_uses_structured_args(self):
        from generator.agent.nodes.api_lookup import api_lookup_node

        class _Retriever:
            def is_available(self):
                return True

            def known_api_names(self):
                return ["DataCopy"]

            def lookup_signature(self, api_name):
                return ApiSignatureResult(
                    api_name=api_name,
                    signature="sig",
                    supported_dtypes=[],
                    repeat_times_limit=None,
                    params=[],
                    example_call="",
                    source_doc="doc.md",
                    details="ok",
                    header_files=["kernel_operator_data_copy_intf.h"],
                    doc_metadata=[{"path": "api-datacopy.md", "title": "DataCopy", "section": "选择规则", "excerpt": "DataCopyPad"}],
                )

        out = api_lookup_node(
            {
                "current_query": "API: signatures",
                "tool_choice_json": {"tool": "api_lookup", "query": "API: signatures", "args": {"api_name": "DataCopy"}},
                "query_round_count": 0,
            },
            _Retriever(),
        )
        self.assertEqual(out["api_lookup_result"]["api_name"], "DataCopy")
        self.assertIn("kernel_operator_data_copy_intf.h", out["api_lookup_results"][0])

    def test_env_check_api_node_uses_structured_args(self):
        from generator.agent.nodes.env_check import env_check_api_node

        class _Retriever:
            def is_available(self):
                return True

            def check_api_exists(self, api_name):
                return ApiCheckResult(
                    found=True,
                    api_name=api_name,
                    header_files=["a.h"],
                    matches=["a.h:1:MatmulType"],
                    summary="ok",
                )

        out = env_check_api_node(
            {
                "current_query": "check api signatures",
                "tool_choice_json": {"tool": "env_check_api", "query": "check api signatures", "args": {"api_name": "MatmulType"}},
                "query_round_count": 0,
            },
            _Retriever(),
        )
        self.assertEqual(out["env_check_api_result"]["api_name"], "MatmulType")

    def test_env_check_npu_node_uses_query_type_and_device(self):
        from generator.agent.nodes.env_check import env_check_npu_node

        class _Retriever:
            def is_available(self):
                return True

            def query_npu_devices(self, device_id=None, query_type="info"):
                return NpuDeviceResult(
                    available=True,
                    query_type=query_type,
                    raw_output=f"device={device_id}, type={query_type}",
                )

        out = env_check_npu_node(
            {
                "current_query": "check NPU memory on device 1",
                "tool_choice_json": {"tool": "env_check_npu", "query": "check NPU memory on device 1", "args": {"query_type": "memory", "device_id": 1}},
                "query_round_count": 0,
            },
            _Retriever(),
        )
        self.assertEqual(out["env_check_npu_result"]["query_type"], "memory")
        self.assertIn("device=1", out["env_check_npu_result"]["raw_output"])

    def test_npu_arch_node_uses_structured_chip_arg(self):
        from generator.agent.nodes.npu_arch import npu_arch_node
        from generator.agent.retrievers.npu_arch_retriever import ChipSpecResult

        class _Retriever:
            def list_chips(self):
                return ["Ascend910B2", "Ascend950DT"]

            def lookup_chip_spec(self, chip_name):
                return ChipSpecResult(
                    chip_name=chip_name,
                    npu_arch="DAV_2201",
                    ub_capacity_bytes=196608,
                    vector_core_num=4,
                    cube_core_num=1,
                    hbm_capacity_gb=64,
                    max_block_dim=65535,
                    max_tile_size={"float": 4096},
                    supported_apis=["Vector"],
                    features=["Pipeline"],
                    arch_compile_macro="DAV_2201",
                    soc_version=chip_name,
                    details="ok",
                )

        out = npu_arch_node(
            {
                "current_query": "chip specs",
                "tool_choice_json": {"tool": "npu_arch", "query": "chip specs", "args": {"chip_name": "Ascend950DT"}},
                "query_round_count": 0,
            },
            _Retriever(),
        )
        self.assertEqual(out["npu_arch_result"]["chip_name"], "Ascend950DT")


class TestEnvCheckApiExactMatching(unittest.TestCase):
    def test_check_api_exists_ignores_identifier_substrings(self):
        with tempfile.TemporaryDirectory() as tmp:
            include_dir = os.path.join(tmp, "aarch64-linux", "ascendc", "act", "include")
            os.makedirs(include_dir, exist_ok=True)
            with open(os.path.join(include_dir, "fake.h"), "w", encoding="utf-8") as handle:
                handle.write("uint64_t maxStepK_ = 0;\n")

            with mock.patch.object(env_checker_mod, "_find_cann_home", return_value=tmp), \
                 mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": ""}, clear=False):
                result = env_checker_mod.check_api_exists("Maxs")

        self.assertFalse(result.found)

    def test_check_api_exists_matches_exact_function_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            include_dir = os.path.join(tmp, "aarch64-linux", "ascendc", "act", "include")
            os.makedirs(include_dir, exist_ok=True)
            with open(os.path.join(include_dir, "fake.h"), "w", encoding="utf-8") as handle:
                handle.write("__aicore__ inline void DataCopy(LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t count);\n")

            with mock.patch.object(env_checker_mod, "_find_cann_home", return_value=tmp), \
                 mock.patch.dict(os.environ, {"ASCEND_OPP_PATH": ""}, clear=False):
                result = env_checker_mod.check_api_exists("AscendC::DataCopy")

        self.assertTrue(result.found)
        self.assertTrue(any("DataCopy" in match for match in result.matches))


class TestEnvCheckNpuRetriever(unittest.TestCase):
    def test_info_query_ignores_device_id_flag(self):
        with mock.patch.object(env_checker_mod, "_check_tool", return_value="npu-smi"), \
             mock.patch.object(env_checker_mod, "_run_cmd", return_value="ok") as run_cmd:
            result = env_checker_mod.query_npu_devices(device_id=0, query_type="info")

        self.assertTrue(result.available)
        run_cmd.assert_called_once_with("npu-smi info", timeout=15.0)

    def test_scoped_query_maps_logical_device_to_card_id(self):
        def _fake_run(cmd: str, timeout: float = 15.0) -> str:
            if cmd == "npu-smi info -m":
                return (
                    "NPU ID                         Chip ID                        Chip Logic ID                  Chip Name\n"
                    "5                              0                              0                              Ascend 910B2\n"
                    "6                              0                              1                              Ascend 910B2\n"
                )
            if cmd == "npu-smi info -t memory -i 5":
                return "HBM Capacity(MB): 65536"
            raise AssertionError(f"unexpected cmd: {cmd}")

        with mock.patch.object(env_checker_mod, "_check_tool", return_value="npu-smi"), \
             mock.patch.object(env_checker_mod, "_run_cmd", side_effect=_fake_run):
            result = env_checker_mod.query_npu_devices(device_id=0, query_type="memory")

        self.assertTrue(result.available)
        self.assertIn("65536", result.raw_output)


class TestCodeSearchSnippetHybridRetriever(unittest.TestCase):
    def _write_file(self, root: str, relative_path: str, content: str) -> None:
        path = os.path.join(root, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def test_keyword_extraction_uses_boundary_matching(self):
        from generator.agent.retrievers.code_search_snippet_retriever import _extract_keywords

        self.assertIn("matmul", _extract_keywords("matmul example"))
        self.assertIn("mul", _extract_keywords("mul example"))
        self.assertNotIn("mul", _extract_keywords("matmul example"))

    def test_host_tiling_query_prefers_host_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/02_features/00_compilation/custom_op/op_host/add_custom/add_custom_tiling.asc",
                """
class AddCustomTiling {
public:
    void InferShape() {
        auto xShape = context->GetInputShape(0);
        context->SetBlockDim(8);
        (void)xShape;
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/02_features/00_compilation/custom_op/op_kernel/add_custom/add_custom_kernel.asc",
                """
class AddCustomKernel {
public:
    __aicore__ inline void Process() {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
        AscendC::DataCopy(xLocal, xGm, 128);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )
            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                matches = retriever.search(
                    query="host tiling GetInputShape InferShape example",
                    source="asc_devkit",
                    top_k=2,
                )

        self.assertTrue(matches)
        self.assertIn("op_host", matches[0].relative_path)

    def test_dense_branch_can_recall_semantic_match_without_lexical_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/pointwise_conv1x1/pointwise_conv1x1.asc",
                """
class PointwiseConvKernel {
public:
    __aicore__ inline void Process() {
        // pointwise conv 1x1 reduction over Cin
        AscendC::TPipe pipe;
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul_example/matmul_example.asc",
                """
class MatmulKernel {
public:
    __aicore__ inline void Process() {
        // constant tiling matmul example
        AscendC::TPipe pipe;
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            def _fake_dense_scores(query: str, records, candidate_indices):
                del query
                scores = {}
                for index in candidate_indices:
                    rel = records[index].relative_path
                    if "pointwise_conv1x1" in rel:
                        scores[index] = 0.95
                    elif "matmul_example" in rel:
                        scores[index] = 0.15
                return scores

            with mock.patch.object(retriever, "_compute_dense_scores", side_effect=_fake_dense_scores):
                matches = retriever.search(
                    query="channel projection mixer",
                    source="asc_devkit",
                    top_k=2,
                )

        self.assertTrue(matches)
        self.assertIn("pointwise_conv1x1", matches[0].relative_path)

    def test_dense_encoding_falls_back_to_cpu_after_npu_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc",
                """
class GeluKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::Gelu(dstLocal, srcLocal, 128);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
                dense_devices=["npu:0", "cpu"],
            )

            seen_devices = []

            class _FakeModel:
                def __init__(self, device: str):
                    self.device = device

                def encode(self, texts, **kwargs):
                    del kwargs
                    seen_devices.append(self.device)
                    if self.device == "npu:0":
                        raise RuntimeError("NPU out of memory")
                    return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

            with mock.patch.object(retriever, "_load_dense_model", side_effect=lambda device=None: _FakeModel(device or "cpu")), \
                 mock.patch.object(retriever, "_release_dense_model"):
                embeddings = retriever._encode_dense_inputs(["dense example"], stage="encoding")

        self.assertIsNotNone(embeddings)
        self.assertEqual(seen_devices, ["npu:0", "cpu"])

    def test_load_records_only_indexes_asc_devkit_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = os.path.join(tmp, "skills")
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")

            self._write_file(
                skills_root,
                "toy_skill/references/environment-setup.md",
                """
```bash
export CANN_PATH=/home/developer/Ascend/cann
ls $CANN_PATH/include/kernel_operator.h
ls $CANN_PATH/lib64/libregister.so
```
""",
            )
            self._write_file(
                skills_root,
                "toy_skill/references/phase1-design.md",
                """
```cpp
task(description = \"phase design\", prompt = \"do review\")
```
""",
            )
            self._write_file(
                skills_root,
                "toy_skill/references/op_kernel.md",
                """
```cpp
class ToyKernel {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
        AscendC::DataCopy(localX, globalX, 128);
    }
};
```
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc",
                """
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
class GeluKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<float> srcLocal;
        AscendC::LocalTensor<float> dstLocal;
        AscendC::Gelu(dstLocal, srcLocal, 128);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root=skills_root,
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            records = retriever._load_records()

        self.assertTrue(records)
        self.assertTrue(all(record.source == "asc_devkit" for record in records))
        self.assertTrue(any("gelu/gelu.asc" in record.relative_path for record in records))
        self.assertFalse(any("toy_skill" in record.relative_path for record in records))

    def test_load_records_drops_non_asc_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul_fused/matmul_fused.asc",
                """
class MatmulFusedKernel {
public:
    __aicore__ inline void Process() {
        AscendC::TPipe pipe;
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul_fused/l2_cache_optimizer.h",
                """
class L2CacheOptimizer {};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            records = retriever._load_records()

        self.assertTrue(records)
        self.assertTrue(all(record.relative_path.endswith(".asc") for record in records))
        self.assertFalse(any(record.relative_path.endswith(".h") for record in records))

    def test_metadata_ignores_license_comment_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc",
                """
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software under the CANN Open Software License Agreement.
 */
class GeluKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<float> srcLocal;
        AscendC::Gelu(dstLocal, srcLocal, 128);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            matches = retriever.search(
                query="gelu activation kernel example",
                source="asc_devkit",
                top_k=1,
            )

        self.assertTrue(matches)
        self.assertIn("AscendC::Gelu", matches[0].metadata.api_symbols)
        self.assertNotIn("Copyright", matches[0].metadata.api_symbols)

    def test_exact_api_symbol_query_prefers_reducemax_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/04_reduce/reducemax/reducemax.asc",
                """
class KernelReduceMax {
public:
    __aicore__ inline void Compute() {
        AscendC::ReduceMax(dstLocal, srcLocal, tmpLocal, 128, true);
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/02_features/03_basic_api/02_memory_vector_compute/block_reduce_min_max_sum/block_reduce_min_max_sum.asc",
                """
class BlockReduceKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::BlockReduceMax(dstLocal, srcLocal, 2, mask, 1, 1, 8);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                matches = retriever.search(
                    query="AscendC ReduceMax Vmax Max elementwise example",
                    source="asc_devkit",
                    top_k=2,
                )

        self.assertTrue(matches)
        self.assertIn("reducemax/reducemax.asc", matches[0].relative_path)

    def test_retrieve_returns_single_full_file_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc",
                """
class GeluKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::TPipe pipe;
        AscendC::LocalTensor<float> srcLocal;
        AscendC::LocalTensor<float> dstLocal;
        AscendC::Gelu(dstLocal, srcLocal, 128);
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                results = retriever.retrieve(
                    query="gelu activation kernel example",
                    source="asc_devkit",
                )

        self.assertEqual(len(results), 1)
        self.assertIn("Top1 Example", results[0])
        self.assertIn("score_note: score is RRF rank fusion", results[0])
        self.assertIn("confidence:", results[0])
        self.assertIn("matched_branches: metadata, bm25", results[0])
        self.assertIn("branch_scores:", results[0])
        self.assertIn("path: 01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc", results[0])
        self.assertIn("artifact_type: kernel", results[0])
        self.assertIn("AscendC::Gelu", results[0])
        self.assertIn("class GeluKernel", results[0])

    def test_search_result_exposes_confidence_and_explanations(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/02_normalization/layernorm/layernorm.asc",
                """
class LayerNormKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::LayerNorm(dstLocal, srcLocal, gammaLocal, betaLocal, 128);
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul/matmul.asc",
                """
class MatmulKernel {
public:
    __aicore__ inline void Process() {
        AscendC::Matmul<int, int, int, int> mm;
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                matches = retriever.search(
                    query="layernorm kernel tiling example",
                    source="asc_devkit",
                    top_k=1,
                )

        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0].confidence, 0.0)
        self.assertIn(matches[0].confidence_label, {"medium", "high"})
        self.assertEqual(matches[0].matched_branches, ("metadata", "bm25"))
        self.assertGreater(matches[0].candidate_score, 0.0)
        self.assertTrue(matches[0].explanation)
        self.assertTrue(any("operator_families matched" in item for item in matches[0].explanation))

    def test_candidate_prefilter_reduces_large_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            for index in range(80):
                if index < 8:
                    content = f"""
class ReduceKernel{index} {{
public:
    __aicore__ inline void Compute() {{
        AscendC::ReduceMax(dstLocal, srcLocal, tmpLocal, 128, true);
    }}
}};
"""
                    relative_path = (
                        f"01_simd_cpp_api/02_features/03_basic_api/02_memory_vector_compute/"
                        f"reduce_family_{index}/reduce_family_{index}.asc"
                    )
                else:
                    content = f"""
class GenericKernel{index} {{
public:
    __aicore__ inline void Process() {{
        AscendC::TPipe pipe;
        AscendC::DataCopy(dstLocal, srcLocal, 128);
    }}
}};
"""
                    relative_path = (
                        f"01_simd_cpp_api/02_features/00_compilation/custom_op/op_kernel/"
                        f"generic_{index}/generic_{index}.asc"
                    )
                self._write_file(asc_root, relative_path, content)

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            records = retriever._load_records()
            intent = _build_query_intent("max pooling 1d sliding window reduce max kernel")
            candidates = retriever._candidate_indices(
                records,
                "asc_devkit",
                intent,
                result_k=3,
            )
            metadata_scores = retriever._metadata_scores(records, intent, candidates)

        self.assertEqual(len(records), 80)
        self.assertLess(len(candidates), len(records))
        self.assertLessEqual(len(candidates), 64)
        self.assertEqual(len(metadata_scores), len(candidates))

    def test_activation_family_filter_keeps_gelu_query_out_of_matrix_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/01_activation/gelu/gelu.asc",
                """
class GeluKernel {
public:
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<float> srcLocal;
        AscendC::LocalTensor<float> dstLocal;
        AscendC::Gelu(dstLocal, srcLocal, 128);
    }
};
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul/matmul.asc",
                """
class MatmulKernel {
public:
    __aicore__ inline void Process(AscendC::TPipe* pipe) {
        AscendC::Matmul<int, int, int, int> mm;
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                matches = retriever.search(
                    query="AscendC::Gelu tiling kernel_operator TPipe example gelu",
                    source="asc_devkit",
                    top_k=2,
                    operator_families=["activation"],
                )

        self.assertTrue(matches)
        self.assertIn("gelu/gelu.asc", matches[0].relative_path)

    def test_source_group_filter_prefers_custom_op_host_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            asc_root = os.path.join(tmp, "examples")
            knowledge_root = os.path.join(tmp, "knowledge")
            self._write_file(
                asc_root,
                "01_simd_cpp_api/02_features/00_compilation/custom_op/op_host/add_custom/add_custom_host.asc",
                """
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}
""",
            )
            self._write_file(
                asc_root,
                "01_simd_cpp_api/03_libraries/00_matrix/matmul/matmul.asc",
                """
class MatmulKernel {
public:
    __aicore__ inline void Process(AscendC::TPipe* pipe) {
        AscendC::Matmul<int, int, int, int> mm;
    }
};
""",
            )

            retriever = CodeSearchSnippetRetriever(
                cann_skills_root="",
                asc_devkit_examples_root=asc_root,
                knowledge_root=knowledge_root,
            )

            with mock.patch.object(retriever, "_compute_dense_scores", return_value={}):
                matches = retriever.search(
                    query="custom operator op_def_registry tiling InferShape kernel_operator example",
                    source="asc_devkit",
                    top_k=2,
                    artifact_types=["host", "tiling"],
                    source_groups=["asc_devkit_custom_op_host"],
                )

        self.assertTrue(matches)
        self.assertIn("op_host", matches[0].relative_path)


class TestParseToolModePlugins(unittest.TestCase):
    def test_parse_builtin_snippect_tool(self):
        m = parse_tool_mode("kb,code_search_snippet")
        self.assertIn("kb", m)
        self.assertIn("code_search_snippet", m)

    def test_parse_builtin_snippet_alias(self):
        m = parse_tool_mode("kb,code_search_snippet")
        self.assertIn("kb", m)
        self.assertIn("code_search_snippet", m)

    def test_parse_kb_and_plugin(self):
        register_tool(
            RegisteredToolSpec(
                name="echo_test",
                display_name="Echo",
                description="Echoes query for tests.",
                parameter_docs="No extra args.",
                handler=_echo_handler,
                examples=['{"tool":"echo_test","query":"ping","args":null}'],
            )
        )
        try:
            m = parse_tool_mode("kb,echo_test")
            self.assertIn("echo_test", m)
            self.assertIn("kb", m)
            s = tool_mode_to_string(m)
            self.assertIn("echo_test", s)
            self.assertIn("kb", s)
        finally:
            get_tool_registry().unregister("echo_test")

    def test_unknown_plugin_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_mode("kb,not_a_real_plugin_xyz")


class TestNormalizeToolChoice(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_tool_choice_name("kb"), "kb")
        self.assertEqual(normalize_tool_choice_name("KB"), "kb")
        self.assertEqual(normalize_tool_choice_name("CODE_RAG"), "code_rag")
        self.assertEqual(normalize_tool_choice_name("code_search_snippet"), "code_search_snippet")
        self.assertEqual(normalize_tool_choice_name("CODE_SEARCH_SNIPPECT"), "code_search_snippet")
        self.assertEqual(normalize_tool_choice_name("CODE_SEARCH_SNIPPET"), "code_search_snippet")
        self.assertEqual(normalize_tool_choice_name("ASCEND_SEARCH"), "ascend_search")
        self.assertEqual(normalize_tool_choice_name("ASCEND_FETCH"), "ascend_fetch")
        self.assertEqual(normalize_tool_choice_name("ANSWER"), "answer")


class TestHasPlugin(unittest.TestCase):
    def test_has_plugin(self):
        m = frozenset({"my_plug"})
        self.assertTrue(has_plugin(m, "MY_PLUG"))


class TestPluginCannotShadowBuiltin(unittest.TestCase):
    def test_register_builtin_name_fails(self):
        with self.assertRaises(ValueError):
            register_tool(
                RegisteredToolSpec(
                    name="kb",
                    display_name="Bad",
                    description="x",
                    parameter_docs="y",
                    handler=lambda s: {},
                )
            )


class TestRegisteredToolSpecUsageGuidance(unittest.TestCase):
    def test_register_persists_usage_guidance(self):
        register_tool(
            RegisteredToolSpec(
                name="guidance_test_tool",
                display_name="GuidanceTest",
                description="Test tool.",
                parameter_docs="Use query.",
                handler=_echo_handler,
                examples=['{"tool":"guidance_test_tool","query":"x","args":null}'],
                usage_guidance="Always set query to a single token.",
            )
        )
        try:
            spec = get_tool_registry().get("guidance_test_tool")
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertEqual(spec.usage_guidance, "Always set query to a single token.")
        finally:
            get_tool_registry().unregister("guidance_test_tool")


class TestChooseToolPromptDynamicTools(unittest.TestCase):
    def test_kb_only_includes_per_tool_block_not_disabled_tools(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import _build_tool_selection_prompt

        class _C:
            class _Chat:
                completions = None

            chat = _Chat()

        register_builtin_tools_for_mode(
            frozenset({"kb"}),
            client=_C(),
            model="dummy",
            kb_retriever=None,
            web_retriever=None,
            code_retriever=None,
            env_retriever=None,
            npu_arch_retriever=None,
            tiling_retriever=None,
            api_retriever=None,
            code_quality_retriever=None,
            kb_shell_retriever=None,
            ascend_search_retriever=None,
            ascend_fetch_retriever=None,
            plugin_snapshot=None,
        )
        try:
            p = _build_tool_selection_prompt("do task", "", frozenset({"kb"}), 0, "")
            self.assertIn("### `kb`", p)
            self.assertIn("Usage guidance:", p)
            self.assertIn("English", p)
            self.assertNotIn("### `web`", p)
            self.assertNotIn("### `code_rag`", p)
            self.assertIn("exactly one", p.lower())
            self.assertIn("thinking", p)
            self.assertIn("Before choosing a tool", p)
        finally:
            get_tool_registry().clear()


class TestAscendNodesPolicy(unittest.TestCase):
    def test_search_non_chinese_query_rejected(self):
        state = {
            "messages": [],
            "current_query": "DataCopy alignment",
            "query_round_count": 0,
        }
        out = ascend_search_node(state)  # type: ignore[arg-type]
        self.assertEqual(out.get("query_round_count"), 1)
        self.assertTrue(out.get("ascend_search_results"))
        self.assertIn("query must contain Chinese", out["ascend_search_results"][0])

    def test_fetch_rejects_url_not_in_whitelist(self):
        state = {
            "messages": [],
            "current_query": "https://example.com/not-allowed",
            "query_round_count": 1,
            "ascend_search_allowed_urls": [
                "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/85RC1alpha001/..."
            ],
        }
        out = ascend_fetch_node(state)  # type: ignore[arg-type]
        self.assertEqual(out.get("query_round_count"), 2)
        self.assertTrue(out.get("ascend_fetch_results"))
        self.assertIn("not in allowed list", out["ascend_fetch_results"][0])


@unittest.skipUnless(
    importlib.util.find_spec("langchain_core") is not None,
    "langchain_core not installed",
)
class TestChooseToolParseFailure(unittest.TestCase):
    def test_burns_round_and_sets_flag(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import choose_tool_node
        from generator.agent.tool_registry import get_tool_registry

        class _Msg:
            content = "not valid json {{{"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        from langchain_core.messages import HumanMessage

        client = _Client()
        try:
            register_builtin_tools_for_mode(
                frozenset({"kb"}),
                client=client,
                model="dummy",
                kb_retriever=None,
                web_retriever=None,
                code_retriever=None,
                env_retriever=None,
                npu_arch_retriever=None,
                tiling_retriever=None,
                api_retriever=None,
                code_quality_retriever=None,
                kb_shell_retriever=None,
                ascend_search_retriever=None,
                ascend_fetch_retriever=None,
                plugin_snapshot=None,
            )
            state = {
                "messages": [HumanMessage(content="do something")],
                "query_round_count": 0,
            }
            out = choose_tool_node(
                state,  # type: ignore[arg-type]
                client,
                "dummy",
                frozenset({"kb"}),
            )
            self.assertTrue(out.get("tool_choice_parse_failed"))
            self.assertEqual(out.get("query_round_count"), 1)
            self.assertEqual(len(out.get("tool_choice_error_log", [])), 1)
            self.assertEqual(out["tool_choice_error_log"][0].get("kind"), "tool_choice_parse_error")
            self.assertEqual(len(out.get("tool_choice_reasoning_log", [])), 1)
            self.assertFalse(out["tool_choice_reasoning_log"][0].get("parsed_ok"))
        finally:
            get_tool_registry().clear()


class TestAgentReportFields(unittest.TestCase):
    def test_build_report_contains_reasoning_logs(self):
        from generator.agent.agent_runner import _build_report

        final_state = {
            "messages": [],
            "reasoning_content": "final-stage-cot",
            "tool_choice_reasoning_log": [
                {
                    "round": 1,
                    "parsed_ok": True,
                    "selected_tool": "kb",
                    "thinking": {
                        "goal": "find API",
                        "missing_info": "signature",
                    },
                    "reasoning_content": "choose kb first",
                }
            ],
        }

        report = _build_report(final_state)
        self.assertEqual(report.get("reasoning_content"), "final-stage-cot")
        self.assertEqual(report.get("final_generation_reasoning_content"), "final-stage-cot")
        trace = report.get("tool_selection_trace", [])
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].get("thinking", {}).get("goal"), "find API")
        self.assertEqual(len(report.get("tool_calls", [])), 0)

    def test_build_report_merges_round_tool_info(self):
        from generator.agent.agent_runner import _build_report

        final_state = {
            "messages": [],
            "reasoning_content": "",
            "tool_choice_reasoning_log": [
                {
                    "round": 1,
                    "parsed_ok": True,
                    "selected_tool": "ascend_search",
                    "args": {"lang": "zh"},
                    "thinking": {
                        "goal": "find docs",
                        "why_tool": "need official reference",
                    },
                    "reasoning_content": "need docs first",
                }
            ],
            "tool_calls_log": [
                {
                    "round": 1,
                    "tool": "ascend_search",
                    "query": "AscendC 激活函数",
                    "args": {"lang": "zh"},
                    "response": "ok",
                }
            ],
        }
        report = _build_report(final_state)
        calls = report.get("tool_calls", [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].get("tool"), "ascend_search")
        self.assertEqual(calls[0].get("args"), {"lang": "zh"})
        self.assertEqual(calls[0].get("tool_choice", {}).get("selected_tool"), "ascend_search")
        self.assertEqual(calls[0].get("tool_choice", {}).get("thinking", {}).get("goal"), "find docs")


@unittest.skipUnless(
    importlib.util.find_spec("langchain_core") is not None,
    "langchain_core not installed",
)
class TestChooseToolArgsNormalization(unittest.TestCase):
    def test_normalizes_api_args_from_query(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import choose_tool_node
        from langchain_core.messages import HumanMessage

        class _Msg:
            content = '{"tool":"env_check_api","query":"check if AscendC::DataCopy exists","args":null}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        client = _Client()
        try:
            register_builtin_tools_for_mode(
                frozenset({"env_check_api"}),
                client=client,
                model="dummy",
                kb_retriever=None,
                web_retriever=None,
                code_retriever=None,
                env_retriever=None,
                npu_arch_retriever=None,
                tiling_retriever=None,
                api_retriever=None,
                code_quality_retriever=None,
                kb_shell_retriever=None,
                ascend_search_retriever=None,
                ascend_fetch_retriever=None,
                plugin_snapshot=None,
            )
            out = choose_tool_node(
                {"messages": [HumanMessage(content="check if AscendC::DataCopy exists")], "query_round_count": 0},
                client,
                "dummy",
                frozenset({"env_check_api"}),
            )
        finally:
            get_tool_registry().clear()

        self.assertEqual(out["tool_choice_json"]["args"], {"api_name": "AscendC::DataCopy"})

    def test_drops_spurious_device_id_for_generic_info_queries(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import choose_tool_node
        from langchain_core.messages import HumanMessage

        class _Msg:
            content = '{"tool":"env_check_npu","query":"summarize available NPUs","args":{"query_type":"info","device_id":0}}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        client = _Client()
        try:
            register_builtin_tools_for_mode(
                frozenset({"env_check_npu"}),
                client=client,
                model="dummy",
                kb_retriever=None,
                web_retriever=None,
                code_retriever=None,
                env_retriever=None,
                npu_arch_retriever=None,
                tiling_retriever=None,
                api_retriever=None,
                code_quality_retriever=None,
                kb_shell_retriever=None,
                ascend_search_retriever=None,
                ascend_fetch_retriever=None,
                plugin_snapshot=None,
            )
            out = choose_tool_node(
                {"messages": [HumanMessage(content="summarize available NPUs")], "query_round_count": 0},
                client,
                "dummy",
                frozenset({"env_check_npu"}),
            )
        finally:
            get_tool_registry().clear()

        self.assertEqual(out["tool_choice_json"]["args"], {"query_type": "info"})

    def test_infers_code_search_snippet_structured_args(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import choose_tool_node
        from langchain_core.messages import HumanMessage

        class _Msg:
            content = '{"tool":"code_search_snippet","query":"Gelu custom operator op_def_registry tiling InferShape AscendC::Gelu kernel_operator example","args":null}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        client = _Client()
        try:
            register_builtin_tools_for_mode(
                frozenset({"code_search_snippet"}),
                client=client,
                model="dummy",
                kb_retriever=None,
                web_retriever=None,
                code_retriever=None,
                env_retriever=None,
                npu_arch_retriever=None,
                tiling_retriever=None,
                api_retriever=None,
                code_quality_retriever=None,
                kb_shell_retriever=None,
                ascend_search_retriever=None,
                ascend_fetch_retriever=None,
                plugin_snapshot=None,
            )
            out = choose_tool_node(
                {"messages": [HumanMessage(content="need gelu host and tiling examples")], "query_round_count": 0},
                client,
                "dummy",
                frozenset({"code_search_snippet"}),
            )
        finally:
            get_tool_registry().clear()

        self.assertEqual(out["tool_choice_json"]["args"]["source"], "asc_devkit")
        self.assertIn("host", out["tool_choice_json"]["args"]["artifact_types"])
        self.assertIn("tiling", out["tool_choice_json"]["args"]["artifact_types"])
        self.assertIn("activation", out["tool_choice_json"]["args"]["operator_families"])
        self.assertIn("asc_devkit_custom_op_host", out["tool_choice_json"]["args"]["source_groups"])

    def test_infers_npu_arch_chip_name_arg(self):
        from generator.agent.builtin_tools import register_builtin_tools_for_mode
        from generator.agent.nodes.choose_tool import choose_tool_node
        from langchain_core.messages import HumanMessage

        class _Msg:
            content = '{"tool":"npu_arch","query":"Ascend910B2 chip specs","args":null}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        client = _Client()
        try:
            register_builtin_tools_for_mode(
                frozenset({"npu_arch"}),
                client=client,
                model="dummy",
                kb_retriever=None,
                web_retriever=None,
                code_retriever=None,
                env_retriever=None,
                npu_arch_retriever=None,
                tiling_retriever=None,
                api_retriever=None,
                code_quality_retriever=None,
                kb_shell_retriever=None,
                ascend_search_retriever=None,
                ascend_fetch_retriever=None,
                plugin_snapshot=None,
            )
            out = choose_tool_node(
                {"messages": [HumanMessage(content="need 910B2 ub and macros")], "query_round_count": 0},
                client,
                "dummy",
                frozenset({"npu_arch"}),
            )
        finally:
            get_tool_registry().clear()

        self.assertEqual(out["tool_choice_json"]["args"]["chip_name"], "Ascend910B2")
