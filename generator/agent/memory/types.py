"""Common types for memory backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RecallResult:
    """Result from a memory recall operation.

    Attributes:
        text: Natural-language context to inject into the agent prompt.
        raw: Optional raw response dict from the backend (for debugging).
        backend: Identifier of the backend that produced this result.
    """

    text: str
    raw: dict[str, Any] | None = None
    backend: str = ""
