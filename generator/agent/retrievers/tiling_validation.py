from typing import Dict, List

from .tiling_classification import normalize_operator_class
from .tiling_common import dtype_bytes
from .tiling_constants import ALIGNMENT_BYTES, DEFAULT_UB_CAPACITY, MAX_REPEAT_TIMES
from .tiling_types import TilingValidationResult


def validate_tiling_params(
    params: Dict,
    chip: str = "DAV_2201",
) -> TilingValidationResult:
    errors: List[str] = []
    warnings_list: List[str] = []
    checks: Dict[str, bool] = {}

    repeat_times = params.get("repeat_times", 0)
    tile_length = params.get("tile_length", 0)
    ub_usage_bytes = params.get("ub_usage_bytes", 0)
    block_num = params.get("block_num", 1)
    dtype = params.get("dtype", "float")
    operator_class = normalize_operator_class(params.get("operator_class"))

    if chip == "DAV_3510":
        ub_capacity = 253952
    else:
        ub_capacity = DEFAULT_UB_CAPACITY

    elem_size = dtype_bytes(dtype)
    alignment_bytes = 32 if operator_class == "reduction" else ALIGNMENT_BYTES
    alignment_name = "32B" if operator_class == "reduction" else "256B"
    elements_per_alignment = alignment_bytes // max(elem_size, 1)

    repeat_ok = repeat_times <= MAX_REPEAT_TIMES and repeat_times > 0
    checks["repeat_times_limit"] = repeat_ok
    if not repeat_ok:
        if repeat_times <= 0:
            errors.append(f"repeat_times = {repeat_times}，必须 > 0")
        else:
            errors.append(f"repeat_times = {repeat_times}，超过硬件限制 {MAX_REPEAT_TIMES}")

    aligned = (tile_length % elements_per_alignment) == 0 if tile_length > 0 else False
    checks["256b_alignment"] = aligned
    if not aligned and tile_length > 0:
        errors.append(
            f"tile_length = {tile_length}，不是 {alignment_name} 对齐的倍数 "
            f"(dtype={dtype}, elements_per_alignment={elements_per_alignment})"
        )

    ub_ok = ub_usage_bytes <= ub_capacity and ub_usage_bytes > 0
    checks["ub_capacity"] = ub_ok
    if not ub_ok:
        if ub_usage_bytes <= 0:
            errors.append(f"ub_usage_bytes = {ub_usage_bytes}，必须 > 0")
        else:
            errors.append(f"ub_usage_bytes = {ub_usage_bytes}，超过 UB 容量 ({ub_capacity}B, chip={chip})")

    max_blocks = 32 if chip in ("DAV_2201", "DAV_1001", "DAV_2002", "DAV_3002") else 64
    block_ok = 1 <= block_num <= max_blocks
    checks["block_count_limit"] = block_ok
    if not block_ok:
        if block_num <= 0:
            errors.append(f"block_num = {block_num}，必须 >= 1")
        else:
            errors.append(f"block_num = {block_num}，超过 chip={chip} 的硬件限制 ({max_blocks})")

    tile_ok = tile_length > 0
    checks["tile_length_positive"] = tile_ok
    if not tile_ok:
        errors.append(f"tile_length = {tile_length}，必须 > 0")

    if repeat_times > 200:
        warnings_list.append(f"repeat_times = {repeat_times}，接近上限 {MAX_REPEAT_TIMES}，建议分批处理")

    ub_pct = (ub_usage_bytes / ub_capacity * 100.0) if ub_capacity > 0 else 0
    if ub_pct > 90:
        warnings_list.append(f"UB 使用率 {ub_pct:.0f}%，接近容量上限")
    elif ub_pct < 30:
        warnings_list.append(f"UB 使用率 {ub_pct:.0f}%，利用率偏低，可增大 tile_length")

    if block_num > 1 and (tile_length * elem_size * block_num) > ub_capacity:
        warnings_list.append("多核场景下总数据量超过单核 UB 容量，需确认分片逻辑")

    return TilingValidationResult(
        status="ok",
        is_valid=len(errors) == 0,
        reason="",
        errors=errors,
        warnings=warnings_list,
        checks=checks,
    )


def build_skipped_validation_result(reason: str) -> TilingValidationResult:
    return TilingValidationResult(
        status="skipped",
        is_valid=None,
        reason=reason,
        errors=[],
        warnings=[],
        checks={},
    )