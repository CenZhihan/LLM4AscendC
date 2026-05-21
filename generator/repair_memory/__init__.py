"""Cross-attempt repair memory: inbox writes, canonical merge, manifest + selection."""

from .paths import get_memory_root, is_repair_memory_enabled, run_slug_from_run_dir
from .merge import merge_run_inbox
from .inject import build_retrieval_block_for_attempt
from .pipeline import maybe_write_repair_memory_after_eval

__all__ = [
    "get_memory_root",
    "is_repair_memory_enabled",
    "run_slug_from_run_dir",
    "merge_run_inbox",
    "build_retrieval_block_for_attempt",
    "maybe_write_repair_memory_after_eval",
]
