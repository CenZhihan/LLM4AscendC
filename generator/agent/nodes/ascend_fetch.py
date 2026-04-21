"""
Ascend online docs fetch node for generator agent.

Policy:
- Fetch exactly one URL per round
- URL must come from historical ascend_search results in current state
- Persist focused fields for downstream use: main_content + code_examples
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..agent_state import GeneratorAgentState
from ..retrievers.ascend_docs_fetch_retriever import AscendDocsFetchRetriever

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _pick_first_url(text: str) -> tuple[str, int]:
    urls = _URL_RE.findall(text or "")
    if not urls:
        return "", 0
    return urls[0], len(urls)


def _compact_code_examples(examples: List[Dict[str, Any]], *, keep: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in examples[:keep]:
        code = (ex.get("code") or "").strip()
        if len(code) > 400:
            code = code[:400] + "..."
        out.append(
            {
                "id": ex.get("id"),
                "language": ex.get("language"),
                "context": ex.get("context"),
                "code": code,
            }
        )
    return out


def ascend_fetch_node(
    state: GeneratorAgentState,
    retriever: AscendDocsFetchRetriever | None = None,
) -> Dict[str, Any]:
    if retriever is None:
        retriever = AscendDocsFetchRetriever()

    query = (state.get("current_query") or "").strip()
    round_num = state.get("query_round_count", 0) + 1

    url, url_count = _pick_first_url(query)
    if not url:
        msg = "[ascend_fetch_error] no URL found in query; provide one URL from previous ascend_search results."
        return {
            "ascend_fetch_results": [msg],
            "query_round_count": round_num,
            "tool_calls_log": [
                {"round": round_num, "tool": "ascend_fetch", "query": query, "response": msg}
            ],
        }

    allowed_urls = set(state.get("ascend_search_allowed_urls") or [])
    if url not in allowed_urls:
        msg = (
            "[ascend_fetch_error] URL is not in allowed list from current session search results; "
            "run ascend_search first or choose a returned URL."
        )
        return {
            "ascend_fetch_results": [msg],
            "query_round_count": round_num,
            "tool_calls_log": [
                {"round": round_num, "tool": "ascend_fetch", "query": query, "response": msg}
            ],
        }

    fetch_result = retriever.fetch(
        url=url,
        extract_code=True,
        extract_tables=False,
        extract_links=False,
        max_content_chars=6000,
        allow_external=False,
    )
    data = fetch_result.get("data") or {}
    main_content = (data.get("main_content") or "").strip()
    if len(main_content) > 2200:
        main_content = main_content[:2200] + "..."
    code_examples = _compact_code_examples(list(data.get("code_examples") or []), keep=3)

    summary = (
        f"[ascend_fetch] title={data.get('title', '')}\n"
        f"url={url}\n"
        f"content_type={data.get('content_type', '')}\n"
        f"main_content:\n{main_content or '[empty]'}\n"
        f"code_examples_count={len(code_examples)}"
    )
    if url_count > 1:
        summary += f"\n[note] detected {url_count} urls in query, fetched only the first one."

    log_entry = {
        "round": round_num,
        "tool": "ascend_fetch",
        "query": query,
        "response": summary,
    }
    structured = {
        "url": url,
        "title": data.get("title", ""),
        "content_type": data.get("content_type", ""),
        "main_content": main_content,
        "code_examples": code_examples,
        "raw_meta": fetch_result.get("meta", {}),
    }
    print(f"[Round {round_num}] tool=ascend_fetch url={url!r}")
    return {
        "ascend_fetch_results": [summary],
        "ascend_fetch_result": structured,
        "query_round_count": round_num,
        "tool_calls_log": [log_entry],
    }
