"""
Ascend online docs search retriever.

Searches hiascend public docs endpoint and returns normalized results.
"""
from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import requests


class AscendDocsSearchRetriever:
    """Retriever for searching Ascend online documentation."""

    def __init__(
        self,
        *,
        base_url: str = "https://www.hiascend.com",
        timeout_seconds: float = 10.0,
        default_lang: str = "zh",
        default_doc_type: str = "DOC",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_lang = default_lang
        self.default_doc_type = default_doc_type
        self.search_endpoint = "/ascendgateway/ascendservice/content/search"
        self.headers = {
            "x-request-type": "machine",
            "Content-Type": "application/json",
            "Referer": self.base_url,
            "User-Agent": "Mozilla/5.0 (compatible; AscendDocsSearchRetriever/1.0)",
        }

    def is_available(self) -> bool:
        """Return True when runtime dependency is present."""
        return requests is not None

    def search(
        self,
        *,
        keyword: str,
        lang: Optional[str] = None,
        doc_type: Optional[str] = None,
        page_num: int = 1,
        page_size: int = 5,
        version_filter: Optional[str] = None,
        sort: int = 1,
        ignore_correction: bool = False,
        search_type: bool = True,
    ) -> Dict[str, Any]:
        """
        Search online docs and return normalized result.

        Returns:
            {
              "success": bool,
              "message": str,
              "data": [normalized result item...],
              "meta": {...}
            }
        """
        t0 = time.perf_counter()
        normalized_lang = (lang or self.default_lang or "zh").strip().lower()
        normalized_doc_type = (doc_type or self.default_doc_type or "DOC").strip().upper()
        version_filter = (version_filter or "").strip() or None

        params_meta: Dict[str, Any] = {
            "keyword": keyword,
            "lang": normalized_lang,
            "doc_type": normalized_doc_type,
            "page_num": page_num,
            "page_size": page_size,
            "version_filter": version_filter,
        }

        validation_error = self._validate_params(
            keyword=keyword,
            lang=normalized_lang,
            doc_type=normalized_doc_type,
            page_num=page_num,
            page_size=page_size,
        )
        if validation_error:
            return self._build_response(
                success=False,
                message=validation_error,
                data=[],
                error_type="param",
                params=params_meta,
                started_at=t0,
            )

        encoded_keyword = quote(base64.b64encode(keyword.strip().encode("utf-8")).decode("utf-8"))
        request_params = {
            "keyword": encoded_keyword,
            "lang": normalized_lang,
            "type": normalized_doc_type,
            "pageNum": page_num,
            "pageSize": page_size,
            "sort": sort,
            "ignoreCorrection": str(ignore_correction).lower(),
            "searchType": str(search_type).lower(),
        }

        try:
            full_url = urljoin(f"{self.base_url}/", self.search_endpoint.lstrip("/"))
            resp = requests.get(
                full_url,
                params=request_params,
                headers=self.headers,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as exc:
            return self._build_response(
                success=False,
                message=f"search request failed: {exc}",
                data=[],
                error_type="network",
                params=params_meta,
                started_at=t0,
            )
        except ValueError as exc:
            return self._build_response(
                success=False,
                message=f"invalid JSON response: {exc}",
                data=[],
                error_type="parse",
                params=params_meta,
                started_at=t0,
            )

        items = self._extract_items(payload)
        normalized = [self._normalize_item(x) for x in items]
        if version_filter:
            vf = version_filter.lower()
            normalized = [x for x in normalized if vf in (x.get("version") or "").lower()]

        if not normalized:
            return self._build_response(
                success=True,
                message="search completed but no matching documents",
                data=[],
                error_type="empty",
                params=params_meta,
                started_at=t0,
            )

        return self._build_response(
            success=True,
            message="search completed",
            data=normalized,
            error_type=None,
            params=params_meta,
            started_at=t0,
        )

    def _validate_params(
        self,
        *,
        keyword: str,
        lang: str,
        doc_type: str,
        page_num: int,
        page_size: int,
    ) -> Optional[str]:
        if not keyword or not keyword.strip():
            return "keyword is required"
        if lang not in {"zh", "en"}:
            return "lang must be one of: zh, en"
        if doc_type not in {"DOC", "API"}:
            return "doc_type must be one of: DOC, API"
        if page_num < 1 or page_num > 100:
            return "page_num must be in [1, 100]"
        if page_size < 1 or page_size > 10:
            return "page_size must be in [1, 10]"
        return None

    def _extract_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        title = (item.get("docTitle") or item.get("title") or "").strip()
        summary = (item.get("docContent") or item.get("summary") or "").strip()
        url = (item.get("docUrl") or item.get("url") or "").strip()
        version = (item.get("version") or "").strip()
        publish_time = (item.get("publishTime") or "").strip()

        if url.startswith("/source/"):
            url = url.replace("/source/", "/document/detail/", 1)
        if url.startswith("/"):
            url = urljoin(f"{self.base_url}/", url.lstrip("/"))

        return {
            "title": title,
            "summary": summary,
            "display_summary": self._truncate(summary, 180),
            "url": url,
            "version": version,
            "publish_time": publish_time,
            "content_type": item.get("content_type") or item.get("type") or "",
            "relevance_score": item.get("relevance_score") or item.get("score") or 0,
        }

    def _build_response(
        self,
        *,
        success: bool,
        message: str,
        data: List[Dict[str, Any]],
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
                "source": "hiascend_search",
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
