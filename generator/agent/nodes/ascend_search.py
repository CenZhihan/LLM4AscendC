"""
Ascend online docs search node for generator agent.

Fixed policy for now:
- lang: zh
- doc_type: DOC
- version_filter: 8.5.0
- query must contain Chinese text
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState
from ..retrievers.ascend_docs_search_retriever import AscendDocsSearchRetriever

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _extract_user_question(state: GeneratorAgentState) -> str:
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return user_msgs[0].content if user_msgs else ""


def _summarize_results(items: List[Dict[str, Any]], *, top_n: int = 6) -> List[str]:
    out: List[str] = []
    for idx, item in enumerate(items[:top_n], start=1):
        title = (item.get("title") or "").strip() or "(untitled)"
        url = (item.get("url") or "").strip()
        version = (item.get("version") or "").strip() or "unknown"
        summary = (item.get("display_summary") or item.get("summary") or "").strip()
        if len(summary) > 220:
            summary = summary[:220] + "..."
        out.append(
            f"[ascend_search#{idx}] title={title}\n"
            f"version={version}\n"
            f"url={url}\n"
            f"summary={summary}"
        )
    return out


def ascend_search_node(
    state: GeneratorAgentState,
    retriever: AscendDocsSearchRetriever | None = None,
) -> Dict[str, Any]:
    """
    Run Ascend docs search with fixed policy.
    """
    if retriever is None:
        retriever = AscendDocsSearchRetriever()

    query = (state.get("current_query") or "").strip()
    if not query:
        query = _extract_user_question(state).strip()

    round_num = state.get("query_round_count", 0) + 1
    if not _contains_cjk(query):
        msg = "[ascend_search_error] query must contain Chinese keywords (lang fixed to zh)."
        return {
            "ascend_search_results": [msg],
            "query_round_count": round_num,
            "tool_calls_log": [
                {"round": round_num, "tool": "ascend_search", "query": query, "response": msg}
            ],
        }

    result = retriever.search(
        keyword=query,
        lang="zh",
        doc_type="DOC",
        page_num=1,
        page_size=5,
        version_filter="8.5.0",
    )
    items = list(result.get("data") or [])
    allowed_urls = [str(x.get("url") or "").strip() for x in items]
    allowed_urls = [u for u in allowed_urls if u]
    summary_lines = _summarize_results(items)

    response = (
        f"success={result.get('success')} message={result.get('message')} "
        f"count={len(items)} allowed_urls={len(allowed_urls)}\n"
        + ("\n\n".join(summary_lines) if summary_lines else "[no documents matched]")
    )
    log_entry = {
        "round": round_num,
        "tool": "ascend_search",
        "query": query,
        "response": response,
    }

    print(f"[Round {round_num}] tool=ascend_search query={query!r} matched={len(items)}")
    return {
        "ascend_search_results": summary_lines if summary_lines else [response],
        "ascend_search_allowed_urls": allowed_urls,
        "ascend_search_result": result,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
