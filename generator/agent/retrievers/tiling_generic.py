from typing import List

from .tiling_classification import normalize_operator_class
from .tiling_common import dtype_bytes
from .tiling_constants import ALIGNMENT_BYTES, DEFAULT_UB_CAPACITY, MAX_REPEAT_TIMES
from .tiling_types import NUMERIC_OK_STATUS, TilingParamsResult


def compute_tiling_params(
    total_elements: int,
    dtype: str,
    op_type: str = "elementwise",
    intermediate_buffers: int = 0,
    ub_capacity_bytes: int = DEFAULT_UB_CAPACITY,
    max_block_num: int = 0,
) -> TilingParamsResult:
    normalized_op_type = normalize_operator_class(op_type)
    elem_size = dtype_bytes(dtype)
    warnings: List[str] = []

    if max_block_num <= 0:
        max_block_num = 32

    block_num = min(max_block_num, max(1, (total_elements + 31) // 32))
    if block_num > total_elements:
        block_num = max(1, total_elements)

    num_per_core = total_elements // block_num
    tail_num_last_core = total_elements % block_num
    if tail_num_last_core == 0:
        tail_num_last_core = num_per_core if num_per_core > 0 else 0

    if normalized_op_type == "reduction":
        buf_count = max(2, intermediate_buffers + 2)
    elif normalized_op_type == "broadcast":
        buf_count = max(3, intermediate_buffers + 2)
    else:
        buf_count = max(2, intermediate_buffers + 2)

    double_buffer_slots = 2
    max_elements_per_tile = ub_capacity_bytes // (elem_size * buf_count * double_buffer_slots)
    elements_per_256b = ALIGNMENT_BYTES // max(elem_size, 1)
    tile_length = (max_elements_per_tile // elements_per_256b) * elements_per_256b
    tile_length = max(tile_length, elements_per_256b)

    if tile_length > num_per_core:
        tile_length = ((num_per_core + elements_per_256b - 1) // elements_per_256b) * elements_per_256b
        tile_length = max(tile_length, elements_per_256b)

    elements_per_repeat = 128 if elem_size <= 2 else 64 if elem_size <= 4 else 32
    repeat_times = max(1, (tile_length + elements_per_repeat - 1) // elements_per_repeat)

    if repeat_times > MAX_REPEAT_TIMES:
        tile_length = MAX_REPEAT_TIMES * elements_per_repeat
        tile_length = (tile_length // elements_per_256b) * elements_per_256b
        tile_length = max(tile_length, elements_per_256b)
        repeat_times = max(1, (tile_length + elements_per_repeat - 1) // elements_per_repeat)
        warnings.append(
            f"tile_length 受 repeatTimes <= {MAX_REPEAT_TIMES} 限制，从 {max_elements_per_tile} 缩减为 {tile_length}"
        )

    ub_usage_bytes = tile_length * elem_size * buf_count * double_buffer_slots
    ub_usage_pct = (ub_usage_bytes / ub_capacity_bytes) * 100.0 if ub_capacity_bytes > 0 else 0.0

    if ub_usage_bytes > ub_capacity_bytes:
        warnings.append(f"UB 使用量 ({ub_usage_bytes}B) 超过容量 ({ub_capacity_bytes}B)")

    if normalized_op_type == "reduction":
        formula_used = (
            f"Reduction 切分: block_num={block_num}, "
            f"tile_length={tile_length} (受 repeatTimes={repeat_times} 限制), "
            f"buf_count={buf_count} (含中间 buffer)"
        )
    elif normalized_op_type == "broadcast":
        formula_used = (
            f"Broadcast 切分: block_num={block_num}, "
            f"tile_length={tile_length} (repeatTimes={repeat_times}), "
            f"buf_count={buf_count} (含广播源 buffer)"
        )
    else:
        formula_used = (
            f"Elementwise 切分: block_num={block_num}, "
            f"tile_length={tile_length} (repeatTimes={repeat_times}), "
            f"double_buffer_slots={double_buffer_slots}"
        )

    constraints_met = repeat_times <= MAX_REPEAT_TIMES and ub_usage_bytes <= ub_capacity_bytes

    return TilingParamsResult(
        status=NUMERIC_OK_STATUS,
        supported=True,
        operator_class=normalized_op_type if normalized_op_type != "unknown" else "elementwise",
        strategy_kind=f"generic_{normalized_op_type if normalized_op_type != 'unknown' else 'elementwise'}",
        reason="",
        required_inputs=[],
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


def compute_generic_tiling_for_class(
    *,
    total_elements: int,
    dtype: str,
    operator_class: str,
    intermediate_buffers: int,
    ub_capacity_bytes: int,
) -> TilingParamsResult:
    result = compute_tiling_params(
        total_elements=total_elements,
        dtype=dtype,
        op_type=operator_class,
        intermediate_buffers=intermediate_buffers,
        ub_capacity_bytes=ub_capacity_bytes,
    )
    result.operator_class = operator_class
    if operator_class == "elementwise":
        result.strategy_kind = "elementwise_contiguous_split"
        result.formula_used = (
            "Elementwise tiling: contiguous multi-core split + UB chunking + double buffer. "
            + (result.formula_used or "")
        )
    elif operator_class == "broadcast":
        result.strategy_kind = "broadcast_simple_expansion"
        result.formula_used = (
            "Broadcast tiling: contiguous output split + extra source buffer for simple one-way expansion. "
            + (result.formula_used or "")
        )
    return result