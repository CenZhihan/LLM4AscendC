from __future__ import annotations

import unittest
from unittest import mock

from langchain_core.messages import HumanMessage

from generator.agent.nodes.choose_tool import choose_tool_node
from generator.agent.nodes.tiling_calc import tiling_calc_node
from generator.agent.nodes.tiling_validate import tiling_validate_node
from generator.agent.retrievers.tiling_retriever import TilingRetriever, classify_operator_for_tiling


class TestTilingRetriever(unittest.TestCase):
    def test_classify_elementwise_relu(self):
        operator_class = classify_operator_for_tiling(
            args={"op_type": "elementwise"},
            state_category="activation",
            state_op_name="relu",
            query="tiling for relu",
        )
        self.assertEqual(operator_class, "elementwise")

    def test_classify_conversion_transpose(self):
        operator_class = classify_operator_for_tiling(
            args={},
            state_category="conversion",
            state_op_name="transpose",
            query="tiling for transpose",
        )
        self.assertEqual(operator_class, "conversion")

    def test_classify_convolution_from_query(self):
        operator_class = classify_operator_for_tiling(
            args={},
            state_category="",
            state_op_name="",
            query="tiling for conv2d forward",
        )
        self.assertEqual(operator_class, "convolution")

    def test_classify_nn_from_normalization_alias(self):
        operator_class = classify_operator_for_tiling(
            args={"op_type": "normalization"},
            state_category="activation",
            state_op_name="layer_norm",
            query="tiling for layer_norm",
        )
        self.assertEqual(operator_class, "nn")

    def test_compute_ok_for_elementwise_relu(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float",
            op_type="elementwise",
            op_name="relu",
            state_category="activation",
            query="tiling for 1024 float elements elementwise",
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "elementwise")
        self.assertIsNotNone(result.tile_length)

    def test_broadcast_without_shapes_is_rejected(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float",
            op_type="broadcast",
            op_name="add_bias_broadcast",
            query="tiling for add bias broadcast",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "broadcast")
        self.assertEqual(result.strategy_kind, "broadcast_shape_required")

    def test_broadcast_onedim_dimension_collapse_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float32",
            op_type="broadcast",
            op_name="add",
            query="tiling for scalar add broadcast",
            input_shapes=[[1024], [1]],
            output_shape=[1024],
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.strategy_kind, "broadcast_onedim_scalar")
        self.assertEqual(result.load_mode, "onedim")

    def test_broadcast_dav2201_static_route_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float32",
            op_type="broadcast",
            op_name="broadcast_add",
            query="tiling for broadcast add",
            input_shapes=[[1, 64], [16, 64]],
            output_shape=[16, 64],
            chip="DAV_2201",
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.strategy_kind, "broadcast_ub_static")
        self.assertEqual(result.load_mode, "ub_static")

    def test_broadcast_dav2201_transport_fallback_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=481,
            dtype="float32",
            op_type="broadcast",
            op_name="broadcast_add",
            query="tiling for broadcast add",
            input_shapes=[[13, 1], [13, 37]],
            output_shape=[13, 37],
            chip="DAV_2201",
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.strategy_kind, "broadcast_transport_axis_last_dummy_fill")
        self.assertEqual(result.load_mode, "transport_dummy_fill")

    def test_broadcast_dav3510_dynamic_ub_route_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=192,
            dtype="float16",
            op_type="broadcast",
            op_name="broadcast_mul",
            query="tiling for broadcast mul",
            input_shapes=[[2, 1, 32], [2, 3, 32]],
            output_shape=[2, 3, 32],
            chip="DAV_3510",
        )
        self.assertEqual(result.status, "planner_ok")
        self.assertEqual(result.strategy_kind, "broadcast_dynamic_ub")
        self.assertEqual(result.load_mode, "dynamic_ub")

    def test_broadcast_dav3510_nddma_with_loop_route_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=128,
            dtype="float32",
            op_type="broadcast",
            op_name="broadcast_add",
            query="tiling for broadcast add",
            input_shapes=[[1, 2, 1, 2, 1, 2, 1], [2, 2, 2, 2, 2, 2, 2]],
            output_shape=[2, 2, 2, 2, 2, 2, 2],
            chip="DAV_3510",
        )
        self.assertEqual(result.status, "planner_ok")
        self.assertEqual(result.strategy_kind, "broadcast_nddma_with_loop")
        self.assertEqual(result.load_mode, "nddma_with_loop")

    def test_layer_norm_returns_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float",
            op_type="normalization",
            op_name="layer_norm",
            query="tiling for layer_norm",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "nn")

    def test_softmax_returns_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float",
            op_type="normalization",
            op_name="softmax",
            query="tiling for softmax over dim",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "nn")

    def test_argmax_returns_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float",
            op_type="reduction",
            op_name="argmax_over_a_dimension",
            query="tiling for argmax reduction",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "reduction")

    def test_reduction_ar_full_load_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=8192,
            dtype="float32",
            op_type="reduction",
            op_name="reduce_sum",
            query="tiling for reduce_sum over last axis",
            input_shape=[128, 64],
            reduction_axes=[1],
            keepdim=False,
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "reduction")
        self.assertEqual(result.strategy_kind, "reduction_ar_full_load")
        self.assertIsNotNone(result.tile_length)

    def test_reduction_ara_full_load_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=32768,
            dtype="float16",
            op_type="reduction",
            op_name="reduce_sum",
            query="tiling for reduce_sum over middle axis",
            input_shape=[8, 32, 128],
            reduction_axes=[1],
            keepdim=True,
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "reduction")
        self.assertEqual(result.strategy_kind, "reduction_ara_full_load")
        self.assertIsNotNone(result.tile_length)

    def test_reduction_ar_col_split_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=6400000,
            dtype="float32",
            op_type="reduction",
            op_name="reduce_sum",
            query="tiling for reduce_sum over large last axis",
            input_shape=[128, 50000],
            reduction_axes=[1],
            keepdim=False,
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "reduction")
        self.assertEqual(result.strategy_kind, "reduction_ar_col_split")
        self.assertIsNotNone(result.tile_length)

    def test_reduction_ara_row_split_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=524288,
            dtype="float32",
            op_type="reduction",
            op_name="reduce_sum",
            query="tiling for reduce_sum over middle axis with split",
            input_shape=[8, 512, 128],
            reduction_axes=[1],
            keepdim=True,
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "reduction")
        self.assertEqual(result.strategy_kind, "reduction_ara_row_split")
        self.assertIsNotNone(result.tile_length)

    def test_reduction_multi_axis_returns_nested_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=5760,
            dtype="float32",
            op_type="reduction",
            op_name="reduce_sum",
            query="tiling for multi axis reduction",
            input_shape=[4, 8, 6, 5, 6],
            reduction_axes=[1, 3],
            keepdim=False,
        )
        self.assertEqual(result.status, "planner_ok")
        self.assertEqual(result.strategy_kind, "reduction_multi_axis_nested")
        self.assertEqual(result.collapsed_pattern, "ARARA")
        self.assertTrue(result.stage_summaries)

    def test_reduction_with_index_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float32",
            op_type="reduction",
            op_name="reduce_max",
            query="tiling for reduction with index",
            input_shape=[16, 64],
            reduction_axes=[1],
            keepdim=False,
            track_index=True,
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertTrue(result.strategy_kind.startswith("reduction_with_index_"))
        self.assertEqual(result.algorithm_kind, "with_index")

    def test_elementwise_without_explicit_size_is_rejected(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=0,
            dtype="float32",
            op_type="elementwise",
            op_name="relu",
            query="tiling for relu",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.strategy_kind, "elementwise_size_required")

    def test_conversion_returns_layout_required_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=1024,
            dtype="float16",
            op_type="conversion",
            op_name="transpose",
            query="tiling for transpose",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "conversion")
        self.assertEqual(result.strategy_kind, "conversion_transpose_layout_required")

    def test_conversion_transpose_partial_support_returns_ok(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=8192,
            dtype="float16",
            op_type="conversion",
            op_name="transpose",
            query="tiling for transpose",
            input_shape=[64, 128],
            output_shape=[128, 64],
            permutation=[1, 0],
        )
        self.assertEqual(result.status, "numeric_ok")
        self.assertEqual(result.operator_class, "conversion")
        self.assertEqual(result.strategy_kind, "conversion_transpose_last_two_dims")
        self.assertIsNotNone(result.tile_length)

    def test_conversion_transpose_partial_support_rejects_unsupported_permutation(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=16384,
            dtype="float16",
            op_type="conversion",
            op_name="transpose",
            query="tiling for transpose",
            input_shape=[4, 8, 16],
            output_shape=[8, 4, 16],
            permutation=[1, 0, 2],
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "conversion")
        self.assertEqual(result.strategy_kind, "conversion_transpose_permutation_unsupported")

    def test_convolution_returns_window_required_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=4096,
            dtype="float16",
            op_type="convolution",
            op_name="conv2d",
            query="tiling for conv2d",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "convolution")
        self.assertEqual(result.strategy_kind, "convolution_window_required")

    def test_random_returns_state_required_unsupported(self):
        retriever = TilingRetriever()
        result = retriever.compute_tiling(
            total_elements=2048,
            dtype="float16",
            op_type="random",
            op_name="dropout",
            query="tiling for dropout random mask",
        )
        self.assertEqual(result.status, "unsupported_without_operator_specific_strategy")
        self.assertEqual(result.operator_class, "random")
        self.assertEqual(result.strategy_kind, "random_state_required")


class TestTilingNodes(unittest.TestCase):
    def test_tiling_calc_uses_state_op_name_when_query_is_generic(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling please",
                "op_name": "layer_norm",
                "category": "normalization",
                "query_round_count": 0,
                "tool_choice_json": {"tool": "tiling_calc", "query": "tiling please", "args": {}},
            }
        )
        self.assertEqual(
            result["tiling_calc_result"]["status"],
            "unsupported_without_operator_specific_strategy",
        )
        self.assertIn("TILING_CALC_SUMMARY", result["tiling_calc_results"][0])
        self.assertIn("status=unsupported_without_operator_specific_strategy", result["tiling_calc_results"][0])

    def test_tiling_calc_formats_stable_summary_for_conversion_support(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling for transpose",
                "op_name": "transpose",
                "category": "conversion",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_calc",
                    "query": "tiling for transpose",
                    "args": {
                        "dtype": "float16",
                        "op_name": "transpose",
                        "op_type": "conversion",
                        "input_shape": [64, 128],
                        "output_shape": [128, 64],
                        "permutation": [1, 0],
                    },
                },
            }
        )
        summary = result["tiling_calc_results"][0]
        self.assertTrue(summary.startswith("TILING_CALC_SUMMARY\nsummary_version=1\n"))
        self.assertIn("status=numeric_ok", summary)
        self.assertIn("operator_class=conversion", summary)
        self.assertIn("strategy_kind=conversion_transpose_last_two_dims", summary)

    def test_tiling_calc_formats_stable_summary_for_reduction_support(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling for reduce_sum over last axis",
                "op_name": "reduce_sum",
                "category": "reduction",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_calc",
                    "query": "tiling for reduce_sum over last axis",
                    "args": {
                        "dtype": "float32",
                        "op_name": "reduce_sum",
                        "op_type": "reduction",
                        "input_shape": [128, 64],
                        "reduction_axes": [1],
                        "keepdim": False,
                    },
                },
            }
        )
        summary = result["tiling_calc_results"][0]
        self.assertIn("status=numeric_ok", summary)
        self.assertIn("operator_class=reduction", summary)
        self.assertIn("strategy_kind=reduction_ar_full_load", summary)

    def test_tiling_calc_formats_stable_summary_for_reduction_split_support(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling for reduce_sum over large last axis",
                "op_name": "reduce_sum",
                "category": "reduction",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_calc",
                    "query": "tiling for reduce_sum over large last axis",
                    "args": {
                        "dtype": "float32",
                        "op_name": "reduce_sum",
                        "op_type": "reduction",
                        "input_shape": [128, 50000],
                        "reduction_axes": [1],
                        "keepdim": False,
                    },
                },
            }
        )
        summary = result["tiling_calc_results"][0]
        self.assertIn("status=numeric_ok", summary)
        self.assertIn("operator_class=reduction", summary)
        self.assertIn("strategy_kind=reduction_ar_col_split", summary)

    def test_tiling_calc_formats_stable_summary_for_broadcast_support(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling for broadcast add",
                "op_name": "broadcast_add",
                "category": "broadcast",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_calc",
                    "query": "tiling for broadcast add",
                    "args": {
                        "dtype": "float32",
                        "op_name": "broadcast_add",
                        "op_type": "broadcast",
                        "input_shapes": [[13, 1], [13, 37]],
                        "output_shape": [13, 37],
                        "chip": "DAV_2201",
                    },
                },
            }
        )
        summary = result["tiling_calc_results"][0]
        self.assertIn("status=numeric_ok", summary)
        self.assertIn("operator_class=broadcast", summary)
        self.assertIn("strategy_kind=broadcast_transport_axis_last_dummy_fill", summary)

    def test_tiling_calc_rejects_query_only_default_workload(self):
        result = tiling_calc_node(
            {
                "current_query": "tiling for relu",
                "op_name": "relu",
                "category": "elementwise",
                "query_round_count": 0,
                "tool_choice_json": {"tool": "tiling_calc", "query": "tiling for relu", "args": {}},
            }
        )
        self.assertEqual(
            result["tiling_calc_result"]["status"],
            "unsupported_without_operator_specific_strategy",
        )
        self.assertIn("strategy_kind=elementwise_size_required", result["tiling_calc_results"][0])

    def test_tiling_validate_skips_unsupported_result(self):
        result = tiling_validate_node(
            {
                "current_query": "validate tiling",
                "tiling_calc_result": {
                    "status": "unsupported_without_operator_specific_strategy",
                    "operator_class": "nn",
                    "reason": "layer_norm requires operator-specific multi-pass reduction tiling",
                },
                "query_round_count": 0,
            }
        )
        self.assertEqual(result["tiling_validate_result"]["status"], "skipped")
        self.assertIn("status=unsupported_without_operator_specific_strategy", result["tiling_validate_result"]["reason"])

    def test_tiling_validate_accepts_reduction_alignment(self):
        result = tiling_validate_node(
            {
                "current_query": "validate reduction tiling",
                "tiling_calc_result": {
                    "status": "numeric_ok",
                    "operator_class": "reduction",
                    "tile_length": 64,
                    "repeat_times": 1,
                    "ub_usage_bytes": 4672,
                    "block_num": 32,
                    "dtype": "float32",
                },
                "query_round_count": 0,
            }
        )
        self.assertEqual(result["tiling_validate_result"]["status"], "ok")
        self.assertTrue(result["tiling_validate_result"]["is_valid"])

    def test_tiling_validate_skips_planner_only_result(self):
        result = tiling_validate_node(
            {
                "current_query": "validate tiling",
                "tiling_calc_result": {
                    "status": "planner_ok",
                    "operator_class": "broadcast",
                    "strategy_kind": "broadcast_dynamic_ub",
                },
                "query_round_count": 0,
            }
        )
        self.assertEqual(result["tiling_validate_result"]["status"], "skipped")
        self.assertIn("status=planner_ok", result["tiling_validate_result"]["reason"])

    def test_tiling_validate_uses_chip_from_args(self):
        result = tiling_validate_node(
            {
                "current_query": "validate tiling chip=DAV_3510",
                "tool_choice_json": {
                    "tool": "tiling_validate",
                    "query": "validate tiling chip=DAV_3510",
                    "args": {
                        "status": "numeric_ok",
                        "chip": "DAV_3510",
                        "operator_class": "elementwise",
                        "tile_length": 64,
                        "repeat_times": 1,
                        "ub_usage_bytes": 200000,
                        "block_num": 32,
                        "dtype": "float32",
                    },
                },
                "query_round_count": 0,
            }
        )
        self.assertEqual(result["tiling_validate_result"]["status"], "ok")
        self.assertTrue(result["tiling_validate_result"]["is_valid"])


class TestChooseToolGuard(unittest.TestCase):
    def test_choose_tool_redirects_after_unsupported_tiling(self):
        class _FakeMessage:
            def __init__(self, content):
                self.content = content
                self.reasoning_content = ""
                self.model_extra = {}

        class _FakeChoice:
            def __init__(self, content):
                self.message = _FakeMessage(content)

        class _FakeResponse:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        class _FakeCompletions:
            def create(self, **kwargs):
                return _FakeResponse('{"tool":"tiling_calc","query":"tiling for layer_norm","args":null}')

        class _FakeChat:
            def __init__(self):
                self.completions = _FakeCompletions()

        class _FakeClient:
            def __init__(self):
                self.chat = _FakeChat()

        with mock.patch(
            "generator.agent.nodes.choose_tool._build_tool_selection_prompt",
            return_value="prompt",
        ):
            out = choose_tool_node(
                {
                    "messages": [HumanMessage(content="implement layer_norm")],
                    "query_round_count": 0,
                    "tool_calls_log": [],
                    "tool_choice_reasoning_log": [],
                    "tiling_calc_result": {
                        "status": "unsupported_without_operator_specific_strategy",
                        "operator_class": "nn",
                    },
                    "op_name": "layer_norm",
                    "category": "normalization",
                    "base_prompt": "implement layer_norm",
                },
                _FakeClient(),
                "fake-model",
                {"tiling_calc", "code_search_snippet"},
            )
        self.assertEqual(out["next_action"], "code_search_snippet")
        self.assertEqual(out["tool_choice_json"]["tool"], "code_search_snippet")


if __name__ == "__main__":
    unittest.main()