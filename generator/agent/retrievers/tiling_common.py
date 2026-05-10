from typing import Any, List

from .tiling_constants import ALIGNMENT_BYTES, DTYPE_BYTES


def normalize_operator_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def normalize_shape(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []

    shape: List[int] = []
    for dim in value:
        try:
            dim_int = int(dim)
        except (TypeError, ValueError):
            return []
        if dim_int <= 0:
            return []
        shape.append(dim_int)
    return shape


def normalize_permutation(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []

    permutation: List[int] = []
    for index in value:
        try:
            permutation.append(int(index))
        except (TypeError, ValueError):
            return []
    return permutation


def normalize_reduction_axes(value: Any, rank: int) -> List[int]:
    if value is None:
        return []

    candidates = [value] if isinstance(value, int) else value
    if not isinstance(candidates, (list, tuple)):
        return []

    axes: List[int] = []
    for axis in candidates:
        try:
            axis_int = int(axis)
        except (TypeError, ValueError):
            return []
        if axis_int < 0:
            axis_int += rank
        if axis_int < 0 or axis_int >= rank:
            return []
        axes.append(axis_int)

    if len(set(axes)) != len(axes):
        return []
    return sorted(axes)


def num_elements(shape: List[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def dtype_bytes(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype.lower(), 4)


def compute_reduce_tmpbuf_size(reduce_extent: int, elem_size: int) -> int:
    per_repeat = ALIGNMENT_BYTES // max(elem_size, 1)
    per_block = 32 // max(elem_size, 1)
    repeats = max(1, (reduce_extent + per_repeat - 1) // per_repeat)
    tmp_buf_size = ((repeats + per_block - 1) // per_block) * per_block * elem_size
    return max(tmp_buf_size, 4096)


def ceil_align(value: int, align: int) -> int:
    if value <= 0:
        return align
    return ((value + align - 1) // align) * align


def compute_core_split(work_units: int, block_cap: int = 32) -> tuple[int, int, int]:
    block_num = min(block_cap, max(1, work_units))
    units_per_core = (work_units + block_num - 1) // block_num
    tail_units = work_units % units_per_core if units_per_core > 0 else 0
    if tail_units == 0 and work_units > 0:
        tail_units = units_per_core
    return block_num, units_per_core, tail_units


def reduction_a0_tile_base(elem_size: int) -> int:
    return max(1, ALIGNMENT_BYTES // max(elem_size, 1))


def candidate_reduction_tile_lengths(a0: int, elem_size: int) -> List[int]:
    if a0 <= 0:
        return []

    base = reduction_a0_tile_base(elem_size)
    if a0 <= base:
        return [a0]

    candidates = [factor * base for factor in range(a0 // base, 0, -1)]
    if a0 not in candidates:
        candidates.insert(0, a0)
    return candidates