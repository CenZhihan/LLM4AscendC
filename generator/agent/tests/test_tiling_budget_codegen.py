from __future__ import annotations

import unittest

from generator.agent.builtin_tools import register_builtin_tools_for_mode
from generator.agent.nodes.answer import _format_retrieved_content
from generator.agent.nodes.tiling_budget_codegen import tiling_budget_codegen_node
from generator.agent.nodes.tiling_validate import tiling_validate_node
from generator.agent.retrievers.tiling_budget_codegen import plan_tiling_budget_codegen
from generator.agent.tool_registry import get_tool_registry


def _sample_request(total_shape=None):
    return {
        "op_name": "fused_add_relu",
        "op_type": "elementwise",
        "dtype": "float16",
        "chip": "DAV_2201",
        "total_shape": total_shape or [524288],
        "input_tensor_count": 2,
        "output_tensor_count": 1,
        "ub_total_bytes": 196608,
        "ub_reserved_bytes": 4096,
        "enable_double_buffer": True,
        "pipeline_stages": [
            {
                "stage_name": "in_x",
                "position": "VECIN",
                "buffer_role": "input",
                "per_tile_elements": 1,
                "depth": 2,
            },
            {
                "stage_name": "in_y",
                "position": "VECIN",
                "buffer_role": "input",
                "per_tile_elements": 1,
                "depth": 2,
            },
            {
                "stage_name": "out_z",
                "position": "VECOUT",
                "buffer_role": "output",
                "per_tile_elements": 1,
                "depth": 2,
            },
            {
                "stage_name": "tmp_relu_mask",
                "position": "VECCALC",
                "buffer_role": "temp",
                "per_tile_elements": 1,
                "depth": 1,
            },
        ],
    }


class TestTilingBudgetCodegenPlanner(unittest.TestCase):
    def test_plan_budget_codegen_returns_alignment_safe_plan(self):
        result = plan_tiling_budget_codegen(_sample_request())

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.supported)
        self.assertEqual(result.operator_class, "elementwise")
        self.assertEqual(result.strategy_kind, "elementwise_contiguous_split")
        self.assertEqual(result.hardware_validation_status, "ok")
        self.assertEqual(result.planning_validation_status, "ok")
        self.assertEqual(result.block_num, result.block_dim)
        self.assertTrue(result.constraints_met)
        self.assertGreater(result.block_dim or 0, 0)
        self.assertGreater(result.tile_length or 0, 0)
        self.assertGreaterEqual(result.loop_count or 0, 1)
        self.assertEqual(result.ub_reserved_bytes, 4096)
        self.assertEqual(result.seed_result["status"], "numeric_ok")
        self.assertEqual(result.seed_result["strategy_kind"], "elementwise_contiguous_split")
        self.assertEqual(len(result.ub_budget_table), 4)
        self.assertEqual(result.ub_budget_table[0].depth, 2)
        self.assertEqual(result.ub_budget_table[1].depth, 2)
        self.assertEqual(result.ub_budget_table[2].depth, 2)
        self.assertEqual(result.ub_budget_table[3].depth, 1)
        self.assertIn("AscendC::TPipe pipe_;", result.init_code)
        self.assertIn("AscendC::TQue<AscendC::TPosition::VECIN, 2> in_x_;", result.init_code)
        self.assertTrue(
            any("double buffer" in text for text in result.strategy_suggestions),
            msg=result.strategy_suggestions,
        )

    def test_plan_budget_codegen_builds_default_stages_and_tail_guidance(self):
        result = plan_tiling_budget_codegen(
            {
                "op_name": "relu",
                "op_type": "elementwise",
                "dtype": "float16",
                "total_shape": [1000],
                "input_tensor_count": 1,
                "output_tensor_count": 1,
                "enable_double_buffer": True,
            }
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.ub_budget_table), 2)
        self.assertTrue(
            any("tail_length is not alignment-safe" in text for text in result.strategy_suggestions),
            msg=result.strategy_suggestions,
        )


    def test_plan_budget_codegen_last_core_covers_non_divisible_workload(self):
        total_elements = 1000
        result = plan_tiling_budget_codegen(
            {
                "op_name": "relu",
                "op_type": "elementwise",
                "dtype": "float16",
                "total_elements": total_elements,
                "input_tensor_count": 1,
                "output_tensor_count": 1,
                "enable_double_buffer": True,
            }
        )

        self.assertEqual(result.status, "ok")
        covered = (result.block_dim - 1) * result.num_per_core + result.last_core_num
        self.assertEqual(covered, total_elements)
        self.assertEqual(result.last_core_num, result.num_per_core + total_elements % result.block_dim)
        self.assertEqual(result.tail_num_last_core, result.last_core_num)


class TestTilingBudgetCodegenNode(unittest.TestCase):
    def test_node_emits_stable_summary_and_structured_result(self):
        result = tiling_budget_codegen_node(
            {
                "current_query": "plan tiling budget for fused_add_relu",
                "op_name": "fused_add_relu",
                "category": "activation",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_budget_codegen",
                    "query": "plan tiling budget for fused_add_relu",
                    "args": _sample_request(),
                },
            }
        )

        summary = result["tiling_budget_codegen_results"][0]
        self.assertTrue(summary.startswith("TILING_BUDGET_CODEGEN_SUMMARY\nsummary_version=1\n"))
        self.assertIn("status=ok", summary)
        self.assertIn("supported=true", summary)
        self.assertIn("strategy_kind=elementwise_contiguous_split", summary)
        self.assertIn("block_num=32", summary)
        self.assertIn("planning_validation_status=ok", summary)
        self.assertIn("hardware_validation_status=ok", summary)

    def test_tiling_validate_accepts_budget_codegen_result(self):
        budget_out = tiling_budget_codegen_node(
            {
                "current_query": "plan tiling budget for fused_add_relu",
                "op_name": "fused_add_relu",
                "category": "activation",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "tiling_budget_codegen",
                    "query": "plan tiling budget for fused_add_relu",
                    "args": _sample_request(),
                },
            }
        )

        validation = tiling_validate_node(
            {
                "current_query": "validate budgeted tiling",
                "tiling_budget_codegen_result": budget_out["tiling_budget_codegen_result"],
                "query_round_count": 0,
            }
        )
        self.assertEqual(validation["tiling_validate_result"]["status"], "ok")
        self.assertTrue(validation["tiling_validate_result"]["is_valid"])


class TestTilingBudgetCodegenIntegration(unittest.TestCase):
    def test_register_builtin_tools_registers_budget_codegen(self):
        registry = get_tool_registry()
        registry.clear()

        register_builtin_tools_for_mode(
            frozenset({"tiling_budget_codegen"}),
            client=None,
            model="fake-model",
        )

        spec = registry.get("tiling_budget_codegen")
        self.assertIsNotNone(spec)
        self.assertIn("budget", spec.description.lower())

    def test_answer_formatter_includes_budget_codegen_section(self):
        text = _format_retrieved_content(
            {
                "tiling_budget_codegen_results": [
                    "TILING_BUDGET_CODEGEN_SUMMARY\nsummary_version=1\nstatus=ok"
                ]
            }
        )
        self.assertIn("[Tiling budget/codegen]", text)


if __name__ == "__main__":
    unittest.main()