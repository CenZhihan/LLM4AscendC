"""Static rules and advisory engines aligned with CANN 8.3.rc baseline."""

from .cann83_constants import (
    ALIGN_BYTES_CAST_DATACOPY_DEFAULT,
    ALIGN_BYTES_COMPARE_BLOCK,
    CANN_BASELINE_VERSION,
    dtype_elem_bytes,
)
from .dtype_policy_engine import analyze_dtype_policy
from .dma_alignment_engine import analyze_dma_alignment

__all__ = [
    "ALIGN_BYTES_CAST_DATACOPY_DEFAULT",
    "ALIGN_BYTES_COMPARE_BLOCK",
    "CANN_BASELINE_VERSION",
    "dtype_elem_bytes",
    "analyze_dtype_policy",
    "analyze_dma_alignment",
]
