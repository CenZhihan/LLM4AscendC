from __future__ import annotations

import sys
import types
import unittest


if "langchain_core.messages" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")

    class _BaseMessage:
        def __init__(self, content: str = ""):
            self.content = content

    class HumanMessage(_BaseMessage):
        pass

    class AIMessage(_BaseMessage):
        pass

    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    langchain_core.messages = messages
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages

if "langgraph.graph" not in sys.modules:
    langgraph = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")

    class MessagesState(dict):
        pass

    graph.MessagesState = MessagesState
    langgraph.graph = graph
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = graph

from generator.agent.builtin_tools import register_builtin_tools_for_mode
from generator.agent.nodes.answer import _format_retrieved_content
from generator.agent.nodes.shape_stride_layout_validator import shape_stride_layout_validator_node
from generator.agent.retrievers.shape_stride_layout_validator import plan_shape_stride_layout_validation
from generator.agent.tool_registry import get_tool_registry


class TestShapeStrideLayoutValidatorPlanner(unittest.TestCase):
    def test_dense_contiguous_request_is_valid(self):
        result = plan_shape_stride_layout_validation(
            {
                "tensor_shape": [2, 16],
                "tensor_stride": [16, 1],
                "movement_direction": "GM_TO_UB",
                "element_dtype": "float16",
            }
        )

        self.assertEqual(result.status, "VALID")
        self.assertEqual(result.layout_class, "DENSE_CONTIGUOUS")
        self.assertTrue(result.is_contiguous)
        self.assertEqual(result.movement_status, "MOVEMENT_LEGAL")
        self.assertEqual(result.hardware_status, "HARDWARE_COMPATIBLE")

    def test_row_regular_noncontiguous_stays_repairable_not_rebuild(self):
        result = plan_shape_stride_layout_validation(
            {
                "tensor_shape": [2, 16],
                "tensor_stride": [20, 1],
                "movement_direction": "GM_TO_UB",
                "element_dtype": "float16",
                "requested_copy_kind": "ROW_WISE",
            }
        )

        self.assertEqual(result.layout_class, "ROW_REGULAR_NONCONTIGUOUS")
        self.assertFalse(result.requires_rebuild)
        self.assertIn(result.status, {"VALID_WITH_WARNING", "VALID"})

    def test_irregular_layout_requires_rebuild(self):
        result = plan_shape_stride_layout_validation(
            {
                "tensor_shape": [2, 4],
                "tensor_stride": [5, 2],
                "movement_direction": "GM_TO_UB",
                "element_dtype": "float16",
            }
        )

        self.assertEqual(result.status, "REBUILD_REQUIRED")
        self.assertEqual(result.layout_class, "IRREGULAR")
        self.assertTrue(result.requires_rebuild)
        self.assertTrue(any(item.type == "REBUILD_LAYOUT" for item in result.suggestions))

    def test_non_aligned_gm_to_ub_row_copy_recommends_pad_copy(self):
        result = plan_shape_stride_layout_validation(
            {
                "tensor_shape": [2, 13],
                "tensor_stride": [13, 1],
                "movement_direction": "GM_TO_UB",
                "element_dtype": "float16",
                "row_count": 2,
                "row_bytes": 26,
                "requested_copy_kind": "ROW_WISE",
            }
        )

        self.assertEqual(result.status, "REPAIRABLE")
        self.assertEqual(result.movement_status, "MOVEMENT_REQUIRES_PAD_COPY")
        self.assertEqual(result.hardware_status, "HARDWARE_ALIGNMENT_WARNING")
        self.assertTrue(any(item.type == "USE_PAD_COPY" for item in result.suggestions))

    def test_ub_to_gm_stride_unit_mismatch_prefers_unit_fix(self):
        result = plan_shape_stride_layout_validation(
            {
                "tensor_shape": [2, 16],
                "tensor_stride": [32, 1],
                "movement_direction": "UB_TO_GM",
                "element_dtype": "float16",
                "src_stride": 32,
                "requested_copy_kind": "ROW_WISE",
            }
        )

        self.assertEqual(result.status, "REPAIRABLE")
        self.assertEqual(result.stride_status, "STRIDE_UNIT_MISMATCH")
        fixes = [item for item in result.suggestions if item.type == "FIX_STRIDE_UNIT"]
        self.assertTrue(fixes)
        self.assertEqual(fixes[0].patch_fields["src_stride"], 1)


class TestShapeStrideLayoutValidatorNode(unittest.TestCase):
    def test_node_emits_stable_summary(self):
        result = shape_stride_layout_validator_node(
            {
                "current_query": "validate GM to UB row copy",
                "query_round_count": 0,
                "tool_choice_json": {
                    "tool": "shape_stride_layout_validator",
                    "query": "validate GM to UB row copy",
                    "args": {
                        "tensor_shape": [2, 13],
                        "tensor_stride": [13, 1],
                        "movement_direction": "GM_TO_UB",
                        "element_dtype": "float16",
                        "row_count": 2,
                        "row_bytes": 26,
                        "requested_copy_kind": "ROW_WISE",
                    },
                },
            }
        )

        summary = result["shape_stride_layout_validator_results"][0]
        self.assertTrue(summary.startswith("SHAPE_STRIDE_LAYOUT_VALIDATOR_SUMMARY\nsummary_version=1\n"))
        self.assertIn("status=REPAIRABLE", summary)
        self.assertIn("movement_status=MOVEMENT_REQUIRES_PAD_COPY", summary)


class TestShapeStrideLayoutValidatorIntegration(unittest.TestCase):
    def test_register_builtin_tools_registers_validator(self):
        registry = get_tool_registry()
        registry.clear()

        register_builtin_tools_for_mode(
            frozenset({"shape_stride_layout_validator"}),
            client=None,
            model="fake-model",
        )

        spec = registry.get("shape_stride_layout_validator")
        self.assertIsNotNone(spec)
        self.assertIn("stride", spec.description.lower())

    def test_answer_formatter_includes_validator_section(self):
        text = _format_retrieved_content(
            {
                "shape_stride_layout_validator_results": [
                    "SHAPE_STRIDE_LAYOUT_VALIDATOR_SUMMARY\nsummary_version=1\nstatus=VALID"
                ]
            }
        )
        self.assertIn("[Shape/stride/layout validator]", text)


if __name__ == "__main__":
    unittest.main()