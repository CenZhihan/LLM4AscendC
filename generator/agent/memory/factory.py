"""Factory for creating memory backends from CLI / config arguments."""

from __future__ import annotations

import pathlib
from typing import Any

from .base import MemoryBackend
from .null import NullMemoryBackend
from .local_repair import LocalRepairMemoryBackend
from .tencentdb import TencentDBMemoryBackend


SUPPORTED_BACKENDS = {"off", "local", "tencentdb"}


def create_memory_backend(
    backend: str,
    *,
    run_dir: pathlib.Path,
    op: str,
    url: str = "http://127.0.0.1:8420",
    session_prefix: str = "llm4ascend",
    recall_limit: int = 5,
    timeout: float = 5.0,
    keep_local_repair_context: bool = True,
) -> MemoryBackend:
    """Instantiate a memory backend by name.

    Args:
        backend: One of ``off``, ``local``, ``tencentdb``.
        run_dir: Operator run directory (contains ``attempt1..N``).
        op: Operator key (e.g. ``"gelu"``).
        url: TencentDB Gateway base URL (used when backend is ``tencentdb``).
        session_prefix: Prefix for session keys (used when backend is ``tencentdb``).
        recall_limit: Max memories to recall (used when backend is ``tencentdb``).
        timeout: HTTP timeout in seconds for REST calls.
        keep_local_repair_context: When ``True`` and backend is ``tencentdb``,
            also read/write local ``repair_context.txt`` as exact evidence.

    Returns:
        An object satisfying the :class:`MemoryBackend` protocol.

    Raises:
        ValueError: If *backend* is not recognised.
    """
    backend = backend.lower().strip()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown memory backend: {backend!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    if backend == "off":
        return NullMemoryBackend()

    if backend == "local":
        return LocalRepairMemoryBackend(run_dir=run_dir, op=op)

    # backend == "tencentdb"
    tencent = TencentDBMemoryBackend(
        url=url,
        timeout=timeout,
        keep_local_repair_context=keep_local_repair_context,
        run_dir=run_dir if keep_local_repair_context else None,
        op=op if keep_local_repair_context else None,
    )
    if not tencent.health_check():
        print(
            f"[WARN] TencentDB memory health check failed (url={url}), "
            f"falling back to local memory backend for op={op}"
        )
        return LocalRepairMemoryBackend(run_dir=run_dir, op=op)
    return tencent
