#!/usr/bin/env python3
"""
Playground script for Ascend online docs tools.

Modes:
  - search: call search retriever only
  - fetch: call fetch retriever only
  - chain: search then fetch top-k / picked indices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from generator.agent.retrievers.ascend_docs_fetch_retriever import AscendDocsFetchRetriever
from generator.agent.retrievers.ascend_docs_search_retriever import AscendDocsSearchRetriever


def _print_search_summary(result: Dict[str, Any]) -> None:
    print(f"[search] success={result.get('success')} message={result.get('message')}")
    meta = result.get("meta") or {}
    print(f"[search] latency_ms={meta.get('latency_ms')} error_type={meta.get('error_type')}")
    items = result.get("data") or []
    print(f"[search] results={len(items)}")
    for i, item in enumerate(items, start=1):
        print(f"  {i}. {item.get('title', '')}")
        print(f"     version={item.get('version', '')} url={item.get('url', '')}")
        print(f"     summary={item.get('display_summary', '')}")


def _print_fetch_summary(result: Dict[str, Any], *, show_main_preview_chars: int) -> None:
    print(f"[fetch] success={result.get('success')} message={result.get('message')}")
    meta = result.get("meta") or {}
    print(f"[fetch] latency_ms={meta.get('latency_ms')} error_type={meta.get('error_type')}")
    data = result.get("data") or {}
    print(f"[fetch] title={data.get('title', '')}")
    print(f"[fetch] content_type={data.get('content_type', '')}")
    print(f"[fetch] api_function={((data.get('api_details') or {}).get('function_name') or '')}")
    print(f"[fetch] code_examples={len(data.get('code_examples') or [])}")
    print(f"[fetch] tables={len(data.get('tables') or [])}")
    print(f"[fetch] links={len(data.get('links') or [])}")
    main_content = data.get("main_content") or ""
    preview = main_content[:show_main_preview_chars]
    if len(main_content) > show_main_preview_chars:
        preview += "..."
    print(f"[fetch] main_content_preview({show_main_preview_chars}): {preview}")


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] wrote JSON to: {target}")


def _chain_pick_urls(search_result: Dict[str, Any], fetch_topk: int, pick_index: str) -> List[str]:
    items = search_result.get("data") or []
    if not items:
        return []

    if pick_index:
        out: List[str] = []
        for token in pick_index.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            if 1 <= idx <= len(items):
                url = (items[idx - 1].get("url") or "").strip()
                if url:
                    out.append(url)
        return out

    out = []
    for item in items[: max(fetch_topk, 0)]:
        url = (item.get("url") or "").strip()
        if url:
            out.append(url)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Ascend docs search/fetch tools before wiring into LLM agent."
    )
    parser.add_argument("--mode", choices=["search", "fetch", "chain"], default="chain")

    # Search params
    parser.add_argument("--keyword", type=str, default="", help="Search keyword for search/chain mode.")
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--doc_type", type=str, default="DOC", choices=["DOC", "API"])
    parser.add_argument("--page_num", type=int, default=1)
    parser.add_argument("--page_size", type=int, default=5)
    parser.add_argument("--version", type=str, default="", help="Version filter, e.g. 8.5.0")

    # Fetch params
    parser.add_argument("--url", type=str, default="", help="URL for fetch mode.")
    parser.add_argument("--extract_code", action="store_true", default=False)
    parser.add_argument("--extract_tables", action="store_true", default=False)
    parser.add_argument("--extract_links", action="store_true", default=False)
    parser.add_argument("--max_content_chars", type=int, default=12000)
    parser.add_argument("--max_code_examples", type=int, default=12)
    parser.add_argument("--max_tables", type=int, default=12)
    parser.add_argument("--max_links", type=int, default=80)
    parser.add_argument("--allow_external", action="store_true", default=False)

    # Chain selection
    parser.add_argument("--fetch_topk", type=int, default=1, help="Top-k URLs to fetch in chain mode.")
    parser.add_argument(
        "--pick_index",
        type=str,
        default="",
        help="Comma-separated 1-based result indices to fetch (overrides --fetch_topk), e.g. 1,3",
    )

    # General output
    parser.add_argument("--print_json", action="store_true", default=False)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--show_main_preview_chars", type=int, default=240)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    searcher = AscendDocsSearchRetriever()
    fetcher = AscendDocsFetchRetriever(
        max_content_chars=args.max_content_chars,
        max_code_examples=args.max_code_examples,
        max_tables=args.max_tables,
        max_links=args.max_links,
    )

    payload: Dict[str, Any] = {"mode": args.mode, "results": {}}

    if args.mode in {"search", "chain"}:
        if not args.keyword.strip():
            raise SystemExit("--keyword is required for search/chain mode")
        search_result = searcher.search(
            keyword=args.keyword,
            lang=args.lang,
            doc_type=args.doc_type,
            page_num=args.page_num,
            page_size=args.page_size,
            version_filter=args.version or None,
        )
        payload["results"]["search"] = search_result
        _print_search_summary(search_result)
    else:
        search_result = {}

    if args.mode == "fetch":
        if not args.url.strip():
            raise SystemExit("--url is required for fetch mode")
        fetch_result = fetcher.fetch(
            url=args.url,
            extract_code=args.extract_code,
            extract_tables=args.extract_tables,
            extract_links=args.extract_links,
            max_content_chars=args.max_content_chars,
            allow_external=args.allow_external,
        )
        payload["results"]["fetch"] = fetch_result
        _print_fetch_summary(fetch_result, show_main_preview_chars=args.show_main_preview_chars)

    elif args.mode == "chain":
        urls = _chain_pick_urls(search_result, fetch_topk=args.fetch_topk, pick_index=args.pick_index)
        chain_fetch_results: List[Dict[str, Any]] = []
        if args.verbose:
            print(f"[chain] selected_urls={len(urls)}")
        for i, url in enumerate(urls, start=1):
            print(f"[chain] fetching {i}/{len(urls)}: {url}")
            fetch_result = fetcher.fetch(
                url=url,
                extract_code=args.extract_code,
                extract_tables=args.extract_tables,
                extract_links=args.extract_links,
                max_content_chars=args.max_content_chars,
                allow_external=args.allow_external,
            )
            chain_fetch_results.append({"url": url, "fetch": fetch_result})
            _print_fetch_summary(fetch_result, show_main_preview_chars=args.show_main_preview_chars)
        payload["results"]["chain_fetch"] = chain_fetch_results

    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.save_json:
        _save_json(args.save_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
