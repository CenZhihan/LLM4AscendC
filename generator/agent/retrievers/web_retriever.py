"""
Web Search Retriever wrapper.

Reuses Agent_kernel's DDGS + trafilatura implementation for web search.
"""
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

# Optional dependencies
try:
    from ddgs import DDGS
    _SEARCH_BACKEND = "ddgs"
except ImportError:
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
        from duckduckgo_search import DDGS
        _SEARCH_BACKEND = "duckduckgo_search"
    except ImportError:
        DDGS = None
        _SEARCH_BACKEND = None

try:
    import requests
except ImportError:
    requests = None

try:
    import trafilatura
except ImportError:
    trafilatura = None


class WebRetriever:
    """
    Wrapper for web search retrieval (DDGS + trafilatura).

    Provides unified interface for web search with content extraction.
    """

    def __init__(
        self,
        max_results: int = 8,
        max_fetch_urls: int = 5,
        fetch_timeout: float = 8.0,
        max_chars_per_url: int = 4000,
    ):
        """
        Initialize web retriever.

        Args:
            max_results: Maximum search results from DDGS
            max_fetch_urls: Maximum URLs to fetch and extract content
            fetch_timeout: Timeout for URL fetching (seconds)
            max_chars_per_url: Maximum characters to extract from each URL
        """
        self.max_results = max_results
        self.max_fetch_urls = max_fetch_urls
        self.fetch_timeout = fetch_timeout
        self.max_chars_per_url = max_chars_per_url

    def is_available(self) -> bool:
        """Check if web search is available (DDGS installed)."""
        return DDGS is not None

    def search(self, query: str) -> List[Dict]:
        """
        Perform web search using DDGS.

        Args:
            query: Search query string

        Returns:
            List of search results with title, snippet, url
        """
        if DDGS is None:
            return [{"title": "", "snippet": f"[请安装搜索包: pip install ddgs] 查询: {query}", "url": ""}]

        try:
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=self.max_results)
                results = list(raw) if raw else []

            out: List[Dict] = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                title = (r.get("title") or r.get("name") or "").strip()
                snippet = (
                    r.get("body")
                    or r.get("snippet")
                    or r.get("description")
                    or ""
                ).strip()
                url = (r.get("href") or r.get("url") or "").strip()
                if not (title or snippet or url):
                    continue
                out.append({"title": title, "snippet": snippet, "url": url})

            return out if out else [{"title": "", "snippet": f"[未返回结果] 查询: {query}", "url": ""}]
        except Exception as e:
            return [{"title": "", "snippet": f"[搜索异常: {e}] 查询: {query}", "url": ""}]

    def _is_bad_url(self, url: str) -> bool:
        """Check if URL is a search/aggregation page."""
        if not url:
            return True
        u = url.strip().lower()
        if not (u.startswith("http://") or u.startswith("https://")):
            return True
        bad_patterns = (
            "bing.com/search",
            "duckduckgo.com/?q=",
            "/search?",
            "/search/",
            "google.com/search",
        )
        return any(p in u for p in bad_patterns)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        if requests is None:
            return None
        try:
            resp = requests.get(
                url,
                timeout=self.fetch_timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            if resp.status_code >= 400:
                return None
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except Exception:
            return None

    def _extract_text(self, html: str) -> Optional[str]:
        """Extract main text from HTML using trafilatura."""
        if not html:
            return None
        if trafilatura is None:
            return None
        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            if not text:
                return None
            cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
            return cleaned or None
        except Exception:
            return None

    def _fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch URL and extract main text."""
        html = self._fetch_html(url)
        if not html:
            return None
        text = self._extract_text(html)
        if not text:
            return None
        return text[:self.max_chars_per_url]

    def _score_relevance(self, query: str, title: str, snippet: str, text: Optional[str]) -> int:
        """Score relevance of result to query."""
        q = (query or "").lower()
        terms = [t for t in re.split(r"[\s,，。;；:：/|]+", q) if len(t) >= 2]
        hay = "\n".join([title or "", snippet or "", text or ""]).lower()
        score = 0
        for t in terms:
            if t in hay:
                score += 1
        if "search" in (title or "").lower():
            score -= 2
        return score

    def _format_result(self, title: str, url: str, extracted: Optional[str], snippet: str) -> str:
        """Format a single result for output."""
        ttl = title.strip() if title else "(无标题)"
        u = url.strip() if url else ""
        if extracted:
            return f"【{ttl}】\nURL: {u}\n正文摘录:\n{extracted}".strip()
        sn = snippet.strip() if snippet else ""
        return f"【{ttl}】\nURL: {u}\n（仅搜索摘要）\n{sn}".strip()

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve web search results with content extraction.

        Args:
            query: Search query string

        Returns:
            List of formatted search results (sorted by relevance)
        """
        raw_results = self.search(query)

        # Filter bad URLs
        candidates: List[Dict] = []
        for r in raw_results:
            url = (r.get("url") or "").strip()
            if url and self._is_bad_url(url):
                continue
            candidates.append(r)
        candidates = candidates[:self.max_fetch_urls]

        # Concurrent fetch and extract
        fetched_text: Dict[str, Optional[str]] = {}
        if requests is not None and trafilatura is not None:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {}
                for r in candidates:
                    url = (r.get("url") or "").strip()
                    if not url:
                        continue
                    futs[ex.submit(self._fetch_and_extract, url)] = url
                for fut in as_completed(futs):
                    url = futs[fut]
                    try:
                        fetched_text[url] = fut.result()
                    except Exception:
                        fetched_text[url] = None

        # Score and format results
        scored_blocks: List[Tuple[int, str]] = []
        for r in candidates:
            title = r.get("title") or ""
            snippet = r.get("snippet") or ""
            url = (r.get("url") or "").strip()
            extracted = fetched_text.get(url) if url else None
            score = self._score_relevance(query, title, snippet, extracted)
            block = self._format_result(title, url, extracted, snippet)
            scored_blocks.append((score, block))

        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored_blocks] if scored_blocks else ["(未找到相关结果)"]


def web_search(query: str, max_results: int = 5) -> List[str]:
    """
    Convenience function for web search.

    Args:
        query: Search query string
        max_results: Maximum results to return

    Returns:
        List of formatted search results
    """
    retriever = WebRetriever(max_results=max_results)
    return retriever.retrieve(query)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Ascend C custom operator development"
    retriever = WebRetriever()
    if retriever.is_available():
        print(f"[INFO] Web search available, querying: {q}")
        for t in retriever.retrieve(q):
            print(f"{t}\n")
    else:
        print("[WARN] Web search not available (pip install ddgs)")