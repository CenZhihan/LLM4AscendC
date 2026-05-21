"""Abstract base (Protocol) for memory backends."""

from __future__ import annotations

from typing import Any, Protocol

from .types import RecallResult


class MemoryBackend(Protocol):
    """Unified interface for agent memory backends.

    Implementations may store/recall from local files, TencentDB Agent Memory,
    Mem0, Zep, Letta, or any other external memory system.
    """

    def recall(
        self,
        *,
        query: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> RecallResult:
        """Recall relevant historical context for the given query.

        Args:
            query: Search/query text describing the current situation.
            session_key: Unique session identifier (e.g. "llm4ascend-gelu").
            metadata: Additional structured metadata (attempt_id, op, category, etc.).
            limit: Maximum number of memory items to recall.

        Returns:
            RecallResult containing the natural-language text to inject.
        """
        ...

    def write(
        self,
        *,
        session_key: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write a turn/experience into memory.

        Args:
            session_key: Unique session identifier.
            user_content: The "user" side of the turn (e.g. error context / repair text).
            assistant_content: The "assistant" side (e.g. generated code).
            metadata: Additional structured metadata.
            messages: Optional full message list for backends that need it.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the backend.

        Optional hook called at the end of an operator run.
        """
        ...
