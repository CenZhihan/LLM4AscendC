"""
CANN 8.3.rc baseline constants used by advisory engines and shared with docs-aligned checks.

Aligned with KB snippets / Ascend docs: Cast/DataCopy paths commonly cite 32-byte alignment
for vector paths; Compare requires 256-byte total extent for the compared tensors.

References (Huawei Ascend CANN 8.3 RC documentation — verified during engine implementation):
- Vector/Cast style APIs: 32B alignment on underlying buffers where documented.
- Compare: extent (count * sizeof(T)) must be 256B-aligned per api_doc_retriever Knowledge parity.
"""
from __future__ import annotations

from typing import Dict

CANN_BASELINE_VERSION = "8.3.rc"

# Typical alignment cited for Cast / contiguous GM↔UB moves in Ascend C guides (use DataCopyPad when not met).
ALIGN_BYTES_CAST_DATACOPY_DEFAULT = 32

# Compare API: total bytes = count * sizeof(T) must be multiple of 256 (see api-doc parity).
ALIGN_BYTES_COMPARE_BLOCK = 256

# Normalized dtype label -> element size in bytes (subset used by kernels / PyTorch parity).
_DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "float": 4,
    "fp32": 4,
    "half": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def dtype_elem_bytes(dtype: str) -> int:
    """Return sizeof(element) for a normalized dtype label; defaults to 4 if unknown."""
    key = (dtype or "").strip().lower()
    return _DTYPE_BYTES.get(key, 4)
