"""Local repair-context memory backend.

Preserves the legacy ``attemptN/<op>_repair_context.txt`` mechanism:
* ``write()`` persists the repair text for the current attempt.
* ``recall()`` reads the previous attempt's repair context.
"""

from __future__ import annotations

import pathlib
from typing import Any

from .types import RecallResult


class LocalRepairMemoryBackend:
    """File-based per-attempt repair context backend.

    This is the default backend (--memory-backend local) and matches
    the historical behavior before external memory integrations.
    """

    def __init__(self, run_dir: pathlib.Path | str, op: str) -> None:
        self.run_dir = pathlib.Path(run_dir)
        self.op = op

    def recall(
        self,
        *,
        query: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> RecallResult:
        """Read the previous attempt's repair context file if it exists."""
        metadata = metadata or {}
        attempt_id = metadata.get("attempt_id", 2)
        prev_path_hint = metadata.get("previous_repair_context_path", "")

        if prev_path_hint:
            p = pathlib.Path(prev_path_hint)
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")
                return RecallResult(text=text, backend="local")

        prev_attempt = attempt_id - 1
        repair_path = self.run_dir / f"attempt{prev_attempt}" / f"{self.op}_repair_context.txt"
        if repair_path.is_file():
            text = repair_path.read_text(encoding="utf-8", errors="replace")
            return RecallResult(text=text, backend="local")
        return RecallResult(text="", backend="local")

    def write(
        self,
        *,
        session_key: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write repair text to ``attemptN/<op>_repair_context.txt``."""
        metadata = metadata or {}
        attempt_id = metadata.get("attempt_id", 1)
        repair_text = metadata.get("repair_text", user_content)
        repair_path = self.run_dir / f"attempt{attempt_id}" / f"{self.op}_repair_context.txt"
        repair_path.parent.mkdir(parents=True, exist_ok=True)
        repair_path.write_text(repair_text, encoding="utf-8")

    def close(self) -> None:
        """No-op for file-based backend."""
        pass
