"""Null / no-op memory backend."""

from __future__ import annotations

from typing import Any

from .types import RecallResult


class NullMemoryBackend:
    """Memory backend that does nothing.

    Used when --memory-backend=off.
    """

    def recall(
        self,
        *,
        query: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> RecallResult:
        return RecallResult(text="", backend="null")

    def write(
        self,
        *,
        session_key: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        pass

    def close(self) -> None:
        pass
