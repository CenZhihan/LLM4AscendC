"""
Ascend online docs content fetch retriever.

Fetches and parses detail page content from hiascend docs.
"""
from __future__ import annotations

import html
import re
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class AscendDocsFetchRetriever:
    """Retriever for fetching and structuring Ascend online doc pages."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        allowed_domains: Optional[Sequence[str]] = None,
        max_content_chars: int = 12000,
        max_code_examples: int = 12,
        max_tables: int = 12,
        max_links: int = 80,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.allowed_domains = tuple(allowed_domains or ("hiascend.com",))
        self.max_content_chars = max_content_chars
        self.max_code_examples = max_code_examples
        self.max_tables = max_tables
        self.max_links = max_links
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AscendDocsFetchRetriever/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        }

    def is_available(self) -> bool:
        return requests is not None and BeautifulSoup is not None

    def fetch(
        self,
        *,
        url: str,
        extract_code: bool = True,
        extract_tables: bool = True,
        extract_links: bool = True,
        max_content_chars: Optional[int] = None,
        allow_external: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        max_chars = max_content_chars or self.max_content_chars
        params_meta = {
            "url": url,
            "extract_code": extract_code,
            "extract_tables": extract_tables,
            "extract_links": extract_links,
            "max_content_chars": max_chars,
            "allow_external": allow_external,
        }

        if not url or not url.strip():
            return self._build_response(
                success=False,
                message="url is required",
                data={},
                error_type="param",
                params=params_meta,
                started_at=t0,
            )

        if max_chars <= 0:
            return self._build_response(
                success=False,
                message="max_content_chars must be > 0",
                data={},
                error_type="param",
                params=params_meta,
                started_at=t0,
            )

        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            return self._build_response(
                success=False,
                message="url must start with http:// or https://",
                data={},
                error_type="param",
                params=params_meta,
                started_at=t0,
            )

        host = (parsed.netloc or "").lower()
        if not allow_external and not self._is_allowed_domain(host):
            return self._build_response(
                success=False,
                message=f"domain not allowed: {host}",
                data={},
                error_type="param",
                params=params_meta,
                started_at=t0,
            )

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
        except requests.exceptions.RequestException as exc:
            return self._build_response(
                success=False,
                message=f"fetch request failed: {exc}",
                data={},
                error_type="network",
                params=params_meta,
                started_at=t0,
            )
        except Exception as exc:
            return self._build_response(
                success=False,
                message=f"failed to parse HTML: {exc}",
                data={},
                error_type="parse",
                params=params_meta,
                started_at=t0,
            )

        try:
            data = {
                "url": url,
                "title": self._extract_title(soup),
                "description": self._extract_description(soup),
                "content_type": self._detect_content_type(soup),
                "main_content": self._truncate(self._extract_main_content(soup), max_chars),
                "api_details": self._extract_api_details(soup),
                "code_examples": self._extract_code_examples(soup) if extract_code else [],
                "tables": self._extract_tables(soup) if extract_tables else [],
                "links": self._extract_links(soup) if extract_links else [],
                "metadata": self._extract_metadata(soup),
            }
        except Exception as exc:
            return self._build_response(
                success=False,
                message=f"failed during structured extraction: {exc}",
                data={},
                error_type="parse",
                params=params_meta,
                started_at=t0,
            )

        return self._build_response(
            success=True,
            message="fetch completed",
            data=data,
            error_type=None,
            params=params_meta,
            started_at=t0,
        )

    def _is_allowed_domain(self, host: str) -> bool:
        return any(host == d or host.endswith(f".{d}") for d in self.allowed_domains)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_elem = soup.find("title")
        if title_elem:
            title = title_elem.get_text().strip()
            return re.sub(r"[-|]\s*昇腾社区$", "", title)
        h1_elem = soup.find("h1")
        return h1_elem.get_text().strip() if h1_elem else ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return self._clean_text(meta_desc["content"])
        first_p = soup.find("p")
        if first_p:
            text = first_p.get_text().strip()
            if len(text) > 20:
                return self._clean_text(text)
        return ""

    def _detect_content_type(self, soup: BeautifulSoup) -> str:
        title = self._extract_title(soup).lower()
        content = soup.get_text().lower()
        if any(k in title or k in content for k in ("api", "接口", "函数", "算子")):
            return "API文档"
        if any(k in title or k in content for k in ("教程", "guide", "入门", "学习")):
            return "教程"
        if any(k in title or k in content for k in ("示例", "example", "demo", "样例")):
            return "示例"
        if any(k in title or k in content for k in ("错误", "故障", "问题", "解决")):
            return "故障排除"
        return "文档"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        selectors = (
            ".main-content",
            ".content",
            ".article-content",
            "#content",
            "main",
            "article",
            ".doc-content",
            ".document-content",
            ".api-content",
            ".api-doc-content",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if not node:
                continue
            self._drop_noise(node)
            text = node.get_text(separator=" ", strip=True)
            if len(text) > 100:
                return self._clean_text(text)

        body = soup.find("body")
        if body:
            self._drop_noise(body)
            return self._clean_text(body.get_text(separator=" ", strip=True))
        return ""

    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, elem in enumerate(soup.find_all(["pre", "code"]), start=1):
            code_text = elem.get_text().strip()
            if len(code_text) < 10:
                continue
            out.append(
                {
                    "id": idx,
                    "language": self._detect_code_language(code_text),
                    "code": code_text,
                    "context": self._get_context(elem),
                }
            )
            if len(out) >= self.max_code_examples:
                break
        return out

    def _extract_api_details(self, soup: BeautifulSoup) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "function_name": "",
            "description": "",
            "parameters": [],
            "return_value": "",
            "usage_example": "",
        }
        for header in soup.find_all(["h1", "h2", "h3"]):
            text = header.get_text().strip()
            if re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*\(", text):
                details["function_name"] = text
                break

        tables = soup.find_all("table")
        for table in tables:
            headers = [th.get_text().strip().lower() for th in table.find_all("th")]
            headers_text = " ".join(headers)
            if "参数" in headers_text or "parameter" in headers_text:
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        name = cells[0].get_text().strip()
                        desc = cells[1].get_text().strip()
                        if name:
                            details["parameters"].append({"name": name, "description": desc})
            elif "返回值" in headers_text or "return" in headers_text:
                rows = table.find_all("tr")[1:]
                if rows:
                    cells = rows[0].find_all("td")
                    if cells:
                        details["return_value"] = cells[0].get_text().strip()
        return details

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        tables_out: List[Dict[str, Any]] = []
        for idx, table in enumerate(soup.find_all("table"), start=1):
            table_data: Dict[str, Any] = {"id": idx, "headers": [], "rows": []}
            headers = table.find_all("th")
            if headers:
                table_data["headers"] = [th.get_text().strip() for th in headers]
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells or all(c.name == "th" for c in cells):
                    continue
                table_data["rows"].append([c.get_text(strip=True) for c in cells])
            if table_data["headers"] or table_data["rows"]:
                tables_out.append(table_data)
            if len(tables_out) >= self.max_tables:
                break
        return tables_out

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        links: List[Dict[str, Any]] = []
        for idx, a_tag in enumerate(soup.find_all("a"), start=1):
            href = (a_tag.get("href") or "").strip()
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue
            links.append(
                {
                    "id": idx,
                    "text": a_tag.get_text(strip=True),
                    "href": href,
                    "title": a_tag.get("title", ""),
                    "context": self._get_context(a_tag),
                }
            )
            if len(links) >= self.max_links:
                break
        return links

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")
            if name and content:
                meta[name] = content
        return meta

    def _detect_code_language(self, code_text: str) -> str:
        if re.search(r"#include\s*<", code_text):
            return "C++"
        if re.search(r"def\s+\w+\s*\(|import\s+\w+", code_text):
            return "Python"
        if re.search(r"function\s+\w+\s*\(|var\s+\w+\s*=", code_text):
            return "JavaScript"
        if re.search(r"public\s+class|private\s+\w+", code_text):
            return "Java"
        return "Unknown"

    def _get_context(self, elem: Any) -> str:
        parent = elem.find_parent(["p", "li", "td", "th", "div"])
        if not parent:
            return ""
        text = parent.get_text(" ", strip=True)
        return self._truncate(text, 180)

    def _drop_noise(self, node: Any) -> None:
        for elem in node.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "form", "button"]
        ):
            elem.decompose()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_response(
        self,
        *,
        success: bool,
        message: str,
        data: Dict[str, Any],
        error_type: Optional[str],
        params: Dict[str, Any],
        started_at: float,
    ) -> Dict[str, Any]:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "success": success,
            "message": message,
            "data": data,
            "meta": {
                "source": "hiascend_fetch",
                "error_type": error_type,
                "latency_ms": latency_ms,
                "params": params,
            },
        }

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
