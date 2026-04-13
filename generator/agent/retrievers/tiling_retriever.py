"""
Tiling Retriever for Ascend C kernel development agent.

Provides tiling parameter computation and validation based on hardware
constraints (repeatTimes <= 255, 256B alignment, UB capacity).
"""
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# Structured result types
# ============================================================

@dataclass
class TilingParamsResult:
    """Result of tiling parameter computation."""
    block_num: int
    num_per_core: int
    tail_num_last_core: int
    tile_length: int                        # Elements per tile (one repeat call)
    repeat_times: int                       # Vector compute repeat count
    ub_usage_bytes: int                     # UB memory used
    ub_usage_pct: float                     # UB usage percentage
    formula_used: str                       # Description of formula used
    constraints_met: bool                   # Whether all constraints are met
    warnings: List[str] = field(default_factory=list)


@dataclass
class TilingValidationResult:
    """Result of tiling parameter validation."""
    is_valid: bool
    errors: List[str]                       # Blocking errors
    warnings: List[str]                     # Performance warnings
    checks: Dict[str, bool]                 # {check_name: passed}


# ============================================================
# DType helpers
# ============================================================

_DTYPE_BYTES: Dict[str, int] = {
    "float": 4,
    "float32": 4,
    "half": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int16": 2,
    "short": 2,
    "int8": 1,
    "uint8": 1,
    "char": 1,
    "double": 8,
    "float64": 8,
}

_ALIGNMENT_BYTES = 256
_MAX_REPEAT_TIMES = 255
_DEFAULT_UB_CAPACITY = 196608  # 192 KB (A2/A3)


def _dtype_bytes(dtype: str) -> int:
    """Get byte size for a dtype string."""
    return _DTYPE_BYTES.get(dtype.lower(), 4)


# ============================================================
# Tiling computation engine
# ============================================================

def compute_tiling_params(
    total_elements: int,
    dtype: str,
    op_type: str = "elementwise",
    intermediate_buffers: int = 0,
    ub_capacity_bytes: int = _DEFAULT_UB_CAPACITY,
    max_block_num: int = 0,
) -> TilingParamsResult:
    """
    Compute optimal tiling parameters for an operator.

    Args:
        total_elements: Total number of elements to process
        dtype: Data type string (e.g., "float", "half", "int8")
        op_type: Operator type ("elementwise" / "reduce" / "broadcast")
        intermediate_buffers: Number of intermediate UB buffers needed
        ub_capacity_bytes: UB capacity in bytes (default 192KB for A2/A3)
        max_block_num: Maximum number of blocks (0 = auto based on hardware)

    Returns:
        TilingParamsResult with computed parameters
    """
    elem_size = _dtype_bytes(dtype)
    warnings: List[str] = []

    # === Step 1: Determine block_num (multi-core split) ===
    if max_block_num <= 0:
        max_block_num = 32  # Default for A2/A3

    block_num = min(max_block_num, max(1, (total_elements + 31) // 32))
    # Ensure each block has at least 1 element
    if block_num > total_elements:
        block_num = max(1, total_elements)

    num_per_core = total_elements // block_num
    tail_num_last_core = total_elements % block_num
    if tail_num_last_core == 0:
        tail_num_last_core = num_per_core if num_per_core > 0 else 0

    # === Step 2: Compute tile_length (UB split) ===
    # For elementwise: need 1 input buffer + 1 output buffer in UB
    # For reduce: need input + output + work buffers
    if op_type == "reduce":
        buf_count = max(2, intermediate_buffers + 2)
    elif op_type == "broadcast":
        buf_count = max(3, intermediate_buffers + 2)
    else:
        buf_count = max(2, intermediate_buffers + 2)

    # Each element needs elem_size bytes per buffer (input + output)
    # For elementwise with double buffer: 2 * buf_count slots
    double_buffer_slots = 2
    max_elements_per_tile = (ub_capacity_bytes) // (elem_size * buf_count * double_buffer_slots)

    # Align tile_length to 256B boundary
    elements_per_256b = _ALIGNMENT_BYTES // max(elem_size, 1)
    tile_length = (max_elements_per_tile // elements_per_256b) * elements_per_256b
    tile_length = max(tile_length, elements_per_256b)

    # Clamp to num_per_core if smaller
    if tile_length > num_per_core:
        tile_length = ((num_per_core + elements_per_256b - 1) // elements_per_256b) * elements_per_256b
        tile_length = max(tile_length, elements_per_256b)

    # === Step 3: Compute repeat_times ===
    # For vector compute APIs, each repeat processes a fixed number of elements
    # depending on dtype. E.g., for half: 1 repeat = 128 elements
    elements_per_repeat = 128 if elem_size <= 2 else 64 if elem_size <= 4 else 32

    repeat_times = max(1, (tile_length + elements_per_repeat - 1) // elements_per_repeat)

    if repeat_times > _MAX_REPEAT_TIMES:
        # Reduce tile_length to fit repeat_times limit
        tile_length = _MAX_REPEAT_TIMES * elements_per_repeat
        tile_length = (tile_length // elements_per_256b) * elements_per_256b
        tile_length = max(tile_length, elements_per_256b)
        repeat_times = max(1, (tile_length + elements_per_repeat - 1) // elements_per_repeat)
        warnings.append(
            f"tile_length 受 repeatTimes <= {_MAX_REPEAT_TIMES} 限制，"
            f"从 {max_elements_per_tile} 缩减为 {tile_length}"
        )

    # === Step 4: Compute UB usage ===
    ub_usage_bytes = tile_length * elem_size * buf_count * double_buffer_slots
    ub_usage_pct = (ub_usage_bytes / ub_capacity_bytes) * 100.0 if ub_capacity_bytes > 0 else 0.0

    if ub_usage_bytes > ub_capacity_bytes:
        warnings.append(f"UB 使用量 ({ub_usage_bytes}B) 超过容量 ({ub_capacity_bytes}B)")

    # === Step 5: Determine formula used ===
    if op_type == "reduce":
        formula_used = (
            f"Reduction 切分: block_num={block_num}, "
            f"tile_length={tile_length} (受 repeatTimes={repeat_times} 限制), "
            f"buf_count={buf_count} (含中间 buffer)"
        )
    else:
        formula_used = (
            f"Elementwise 切分: block_num={block_num}, "
            f"tile_length={tile_length} (repeatTimes={repeat_times}), "
            f"double_buffer_slots={double_buffer_slots}"
        )

    constraints_met = repeat_times <= _MAX_REPEAT_TIMES and ub_usage_bytes <= ub_capacity_bytes

    return TilingParamsResult(
        block_num=block_num,
        num_per_core=num_per_core,
        tail_num_last_core=tail_num_last_core,
        tile_length=tile_length,
        repeat_times=repeat_times,
        ub_usage_bytes=ub_usage_bytes,
        ub_usage_pct=round(ub_usage_pct, 1),
        formula_used=formula_used,
        constraints_met=constraints_met,
        warnings=warnings,
    )


def validate_tiling_params(
    params: Dict,
    chip: str = "DAV_2201",
) -> TilingValidationResult:
    """
    Validate tiling parameters against hardware constraints.

    Args:
        params: Dict with tiling parameters (tile_length, repeat_times,
                ub_usage_bytes, block_num, dtype, etc.)
        chip: Chip architecture string (e.g., "DAV_2201", "DAV_3510")

    Returns:
        TilingValidationResult with validation results
    """
    errors: List[str] = []
    warnings_list: List[str] = []
    checks: Dict[str, bool] = {}

    repeat_times = params.get("repeat_times", 0)
    tile_length = params.get("tile_length", 0)
    ub_usage_bytes = params.get("ub_usage_bytes", 0)
    block_num = params.get("block_num", 1)
    dtype = params.get("dtype", "float")

    # UB capacity for this chip
    if chip == "DAV_3510":
        ub_capacity = 253952  # 248 KB
    else:
        ub_capacity = _DEFAULT_UB_CAPACITY  # 192 KB

    elem_size = _dtype_bytes(dtype)
    elements_per_256b = _ALIGNMENT_BYTES // max(elem_size, 1)

    # Check 1: repeat_times <= 255
    repeat_ok = repeat_times <= _MAX_REPEAT_TIMES and repeat_times > 0
    checks["repeat_times_limit"] = repeat_ok
    if not repeat_ok:
        if repeat_times <= 0:
            errors.append(f"repeat_times = {repeat_times}，必须 > 0")
        else:
            errors.append(
                f"repeat_times = {repeat_times}，超过硬件限制 {_MAX_REPEAT_TIMES}"
            )

    # Check 2: 256B alignment
    aligned = (tile_length % elements_per_256b) == 0 if tile_length > 0 else False
    checks["256b_alignment"] = aligned
    if not aligned and tile_length > 0:
        errors.append(
            f"tile_length = {tile_length}，不是 256B 对齐的倍数 "
            f"(dtype={dtype}, elements_per_256b={elements_per_256b})"
        )

    # Check 3: UB capacity
    ub_ok = ub_usage_bytes <= ub_capacity and ub_usage_bytes > 0
    checks["ub_capacity"] = ub_ok
    if not ub_ok:
        if ub_usage_bytes <= 0:
            errors.append(f"ub_usage_bytes = {ub_usage_bytes}，必须 > 0")
        else:
            errors.append(
                f"ub_usage_bytes = {ub_usage_bytes}，超过 UB 容量 ({ub_capacity}B, chip={chip})"
            )

    # Check 4: blockCount limit (A2/A3: max 32, A5: depends)
    max_blocks = 32 if chip in ("DAV_2201", "DAV_1001", "DAV_2002", "DAV_3002") else 64
    block_ok = 1 <= block_num <= max_blocks
    checks["block_count_limit"] = block_ok
    if not block_ok:
        if block_num <= 0:
            errors.append(f"block_num = {block_num}，必须 >= 1")
        else:
            errors.append(
                f"block_num = {block_num}，超过 chip={chip} 的硬件限制 ({max_blocks})"
            )

    # Check 5: tile_length > 0
    tile_ok = tile_length > 0
    checks["tile_length_positive"] = tile_ok
    if not tile_ok:
        errors.append(f"tile_length = {tile_length}，必须 > 0")

    # Warnings
    if repeat_times > 200:
        warnings_list.append(
            f"repeat_times = {repeat_times}，接近上限 {_MAX_REPEAT_TIMES}，"
            f"建议分批处理"
        )

    ub_pct = (ub_usage_bytes / ub_capacity * 100.0) if ub_capacity > 0 else 0
    if ub_pct > 90:
        warnings_list.append(f"UB 使用率 {ub_pct:.0f}%，接近容量上限")
    elif ub_pct < 30:
        warnings_list.append(f"UB 使用率 {ub_pct:.0f}%，利用率偏低，可增大 tile_length")

    if block_num > 1 and (tile_length * elem_size * block_num) > ub_capacity:
        warnings_list.append("多核场景下总数据量超过单核 UB 容量，需确认分片逻辑")

    is_valid = len(errors) == 0

    return TilingValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings_list,
        checks=checks,
    )


# ============================================================
# Retriever class
# ============================================================

class TilingRetriever:
    """
    Tiling parameter computation and validation.

    Provides pure computation based on hardware constraints:
    - repeatTimes <= 255
    - 256B alignment
    - UB capacity (192KB for A2/A3, 248KB for A5)
    - blockCount limit
    """

    def __init__(self):
        pass

    def is_available(self) -> bool:
        """Tiling computation is always available (pure calculation)."""
        return True

    def compute_tiling(
        self,
        total_elements: int,
        dtype: str,
        op_type: str = "elementwise",
        intermediate_buffers: int = 0,
        ub_capacity_bytes: int = _DEFAULT_UB_CAPACITY,
    ) -> TilingParamsResult:
        """
        Compute optimal tiling parameters.

        Args:
            total_elements: Total number of elements to process
            dtype: Data type string
            op_type: Operator type ("elementwise" / "reduce" / "broadcast")
            intermediate_buffers: Number of intermediate UB buffers
            ub_capacity_bytes: UB capacity in bytes

        Returns:
            TilingParamsResult with computed parameters
        """
        return compute_tiling_params(
            total_elements=total_elements,
            dtype=dtype,
            op_type=op_type,
            intermediate_buffers=intermediate_buffers,
            ub_capacity_bytes=ub_capacity_bytes,
        )

    def validate_tiling(
        self,
        tiling_params: Dict,
        chip: str = "DAV_2201",
    ) -> TilingValidationResult:
        """
        Validate tiling parameters against hardware constraints.

        Args:
            tiling_params: Dict with tiling parameters
            chip: Chip architecture string

        Returns:
            TilingValidationResult with validation results
        """
        return validate_tiling_params(tiling_params, chip)
