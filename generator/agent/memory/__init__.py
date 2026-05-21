"""Unified memory backend abstraction for LLM4Ascend Agent."""

from __future__ import annotations

from .base import MemoryBackend
from .types import RecallResult
from .factory import create_memory_backend
from .null import NullMemoryBackend
from .local_repair import LocalRepairMemoryBackend
from .tencentdb import TencentDBMemoryBackend

__all__ = [
    "MemoryBackend",
    "RecallResult",
    "create_memory_backend",
    "NullMemoryBackend",
    "LocalRepairMemoryBackend",
    "TencentDBMemoryBackend",
]
