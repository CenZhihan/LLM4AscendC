"""Unit tests: tool registry, parse_tool_mode with plugins, JSON tool choice."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

from generator.agent.agent_config import (
    parse_tool_mode,
    tool_mode_to_string,
    normalize_tool_choice_name,
    has_plugin,
)
from generator.agent.nodes.ascend_fetch import ascend_fetch_node
from generator.agent.nodes.ascend_search import ascend_search_node
from generator.agent.query_utils import extract_api_name, extract_npu_query_params
from generator.agent.retrievers.api_doc_retriever import ApiDocRetriever, ApiSignatureResult
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


class TestApiRetrieverFallback(unittest.TestCase):
    def test_lookup_signature_falls_back_to_header_search(self):
        retriever = ApiDocRetriever()
        with mock.patch(
            "generator.agent.retrievers.env_checker.check_api_exists",
            return_value=ApiCheckResult(
                found=True,
                api_name="MatmulType",
                header_files=["matmul.h"],
                matches=["matmul.h:42:using MatmulType = int;"],
                summary="found in headers",
            ),
        ):
            result = retriever.lookup_signature("MatmulType")
        self.assertEqual(result.api_name, "MatmulType")
        self.assertIn("MatmulType", result.signature)
        self.assertIn("found in headers", result.details)


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


class TestParseToolModePlugins(unittest.TestCase):
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
        finally:
            get_tool_registry().clear()
