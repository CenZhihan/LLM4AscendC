"""TencentDB Agent Memory REST Gateway backend."""

from __future__ import annotations

import pathlib
from typing import Any

from .types import RecallResult
from .local_repair import LocalRepairMemoryBackend

# Lazy import requests to avoid hard dependency at import time
try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class TencentDBMemoryBackend:
    """Memory backend that talks to the TencentDB Agent Memory Gateway.

    Endpoints used:
      * POST /recall   — fetch long-term memory context
      * POST /capture  — persist a conversation turn
      * GET  /health   — health check (used for graceful fallback)

    When ``keep_local_repair_context`` is *True* (default), this backend
    also reads/writes the local ``attemptN/<op>_repair_context.txt`` files
    so the agent still receives exact per-attempt error evidence.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        keep_local_repair_context: bool = True,
        run_dir: pathlib.Path | str | None = None,
        op: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.keep_local = keep_local_repair_context
        self.local: LocalRepairMemoryBackend | None = None
        if keep_local_repair_context and run_dir is not None and op is not None:
            self.local = LocalRepairMemoryBackend(run_dir, op)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def health_check(self) -> bool:
        """Return ``True`` if the gateway is reachable and healthy."""
        if requests is None:
            return False
        try:
            resp = requests.get(
                f"{self.url}/health",
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            return data.get("status") in ("ok", "degraded")
        except Exception:
            return False

    def recall(
        self,
        *,
        query: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> RecallResult:
        """Recall long-term memory and optionally prepend local exact context."""
        metadata = metadata or {}

        # 1. Local exact repair context (previous attempt)
        local_text = ""
        if self.local is not None:
            local_res = self.local.recall(
                query=query,
                session_key=session_key,
                metadata=metadata,
                limit=limit,
            )
            local_text = local_res.text

        # 2. TencentDB long-term recall
        tencent_text = ""
        if requests is not None:
            try:
                resp = requests.post(
                    f"{self.url}/recall",
                    json={"query": query, "session_key": session_key},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    tencent_text = resp.json().get("context", "")
            except Exception:
                pass

        # 3. Merge: exact local evidence first, then long-term memory
        combined = local_text
        if tencent_text:
            if combined:
                combined += f"\n\nLong-term memory recall:\n{tencent_text}"
            else:
                combined = f"Long-term memory recall:\n{tencent_text}"

        return RecallResult(
            text=combined,
            backend="tencentdb",
            raw={"local": local_text, "tencent": tencent_text},
        )

    def write(
        self,
        *,
        session_key: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist to local repair context (optional) and to TencentDB Gateway."""
        metadata = metadata or {}

        # 1. Local exact repair context
        if self.local is not None:
            self.local.write(
                session_key=session_key,
                user_content=user_content,
                assistant_content=assistant_content,
                metadata=metadata,
                messages=messages,
            )

        # 2. TencentDB capture
        if requests is not None:
            try:
                payload: dict[str, Any] = {
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "session_key": session_key,
                }
                if messages:
                    payload["messages"] = messages
                requests.post(
                    f"{self.url}/capture",
                    json=payload,
                    timeout=self.timeout,
                )
            except Exception:
                pass

    def close(self) -> None:
        """No-op for REST backend; connections are stateless."""
        pass
