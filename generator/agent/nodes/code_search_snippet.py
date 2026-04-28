"""Structural block retrieval node for curated asc-devkit Ascend C sources.

Now unified with semantic fallback via code_rag (Plan Phase 2).
"""
import time
from typing import Any, Dict, Optional

from ..agent_state import GeneratorAgentState
from ..retrievers.code_search_snippet_retriever import CodeSearchSnippetRetriever


def code_search_snippet_node(
    state: GeneratorAgentState,
    retriever: CodeSearchSnippetRetriever = None,
    code_rag_retriever: Optional[Any] = None,
) -> Dict[str, Any]:
    if retriever is None:
        retriever = CodeSearchSnippetRetriever(code_rag_retriever=code_rag_retriever)

    op_name = state.get("op_name", "")
    category = state.get("category", "")
    current_query = (state.get("current_query") or "").strip()
    tool_choice = state.get("tool_choice_json") or {}
    args = tool_choice.get("args") or {}
    source = str(args.get("source") or "asc_devkit")
    artifact_types = args.get("artifact_types") or []
    operator_families = args.get("operator_families") or []
    if not operator_families and args.get("operator_family"):
        operator_families = [args.get("operator_family")]
    source_groups = args.get("source_groups") or []
    context_type = args.get("context_type") or None
    api_patterns = args.get("api_patterns") or []

    query = retriever.build_query(op_name, category, current_query if current_query else None)
    round_num = state.get("query_round_count", 0) + 1
    start_time = time.time()

    print(f"[Round {round_num}] tool=code_search_snippet start, query=\"{query[:100]}...\"")

    if retriever.is_available():
        results = retriever.retrieve(
            query=query,
            source=source if source in {"all", "asc_devkit"} else "asc_devkit",
            artifact_types=artifact_types,
            operator_families=operator_families,
            source_groups=source_groups,
            context_type=context_type,
            api_patterns=api_patterns if api_patterns else None,
        )
    else:
        sources = retriever.available_sources()
        results = [
            "[code_search_snippet] unavailable. "
            f"manifest={sources['manifest_path']}, status={sources['asc_devkit']}"
        ]

    elapsed = time.time() - start_time
    response = "\n".join(results) if results else ""
    log_entry = {
        "round": round_num,
        "tool": "code_search_snippet",
        "query": query,
        "args": args if isinstance(args, dict) else {},
        "response": response,
    }

    print(f"[Round {round_num}] tool=code_search_snippet done in {elapsed:.2f}s, query=\"{query[:100]}...\"")

    return {
        "code_search_snippet_results": results,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
