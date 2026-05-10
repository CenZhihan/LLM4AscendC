from __future__ import annotations

from typing import Any, List, Optional

from .tiling_common import ceil_align, dtype_bytes, normalize_shape, normalize_operator_text, num_elements
from .tiling_constants import ALIGNMENT_BYTES, DEFAULT_UB_CAPACITY, MAX_REPEAT_TIMES
from .tiling_types import NUMERIC_OK_STATUS, PLANNER_OK_STATUS, TilingParamsResult
from .tiling_unsupported import build_unsupported_tiling_result


def _normalize_chip(chip: Any) -> str:
    text = normalize_operator_text(chip).upper()
    if not text:
        return "DAV_2201"
    if "3510" in text or "950" in text:
        return "DAV_3510"
    return "DAV_2201"


def _normalize_input_shapes(input_shapes: Any, input_shape: Any) -> List[List[int]]:
    if isinstance(input_shapes, (list, tuple)) and input_shapes:
        normalized_shapes: List[List[int]] = []
        for item in input_shapes:
            normalized = normalize_shape(item)
            if not normalized:
                return []
            normalized_shapes.append(normalized)
        return normalized_shapes

    normalized = normalize_shape(input_shape)
    return [normalized] if normalized else []


def _pad_shape(shape: List[int], rank: int) -> List[int]:
    return [1] * (rank - len(shape)) + list(shape)


def _infer_output_shape(input_shapes: List[List[int]], output_shape: List[int]) -> List[int]:
    if not input_shapes:
        return []

    rank = max([len(shape) for shape in input_shapes] + ([len(output_shape)] if output_shape else [0]))
    padded_inputs = [_pad_shape(shape, rank) for shape in input_shapes]
    padded_output = _pad_shape(output_shape, rank) if output_shape else None
    inferred: List[int] = []

    for axis in range(rank):
        dims = [shape[axis] for shape in padded_inputs]
        out_dim = max(dims)
        if padded_output is not None:
            out_dim = padded_output[axis]
        if out_dim <= 0:
            return []
        if any(dim not in (1, out_dim) for dim in dims):
            return []
        inferred.append(out_dim)
    return inferred


def _contiguous_strides(shape: List[int]) -> List[int]:
    if not shape:
        return []
    strides = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = running
        running *= shape[index]
    return strides


def _collapse_broadcast_shapes(
    input_shapes: List[List[int]],
    output_shape: List[int],
) -> tuple[List[List[int]], List[int], List[List[int]], List[int], List[List[int]]]:
    rank = len(output_shape)
    padded_inputs = [_pad_shape(shape, rank) for shape in input_shapes]

    collapsed_inputs: List[List[int]] = [[] for _ in padded_inputs]
    collapsed_output: List[int] = []
    collapsed_flags: List[List[int]] = []

    for axis in range(rank):
        flags = [1 if padded_inputs[input_index][axis] == 1 and output_shape[axis] > 1 else 0 for input_index in range(len(padded_inputs))]
        if collapsed_flags and flags == collapsed_flags[-1]:
            collapsed_output[-1] *= output_shape[axis]
            for input_index, padded_shape in enumerate(padded_inputs):
                collapsed_inputs[input_index][-1] *= padded_shape[axis]
            continue

        collapsed_output.append(output_shape[axis])
        collapsed_flags.append(flags)
        for input_index, padded_shape in enumerate(padded_inputs):
            collapsed_inputs[input_index].append(padded_shape[axis])

    input_strides: List[List[int]] = []
    for input_shape in collapsed_inputs:
        base_strides = _contiguous_strides(input_shape)
        input_strides.append(
            [0 if input_shape[axis] == 1 and collapsed_output[axis] > 1 else base_strides[axis] for axis in range(len(collapsed_output))]
        )
    output_strides = _contiguous_strides(collapsed_output)
    return collapsed_inputs, collapsed_output, input_strides, output_strides, collapsed_flags


def _elements_per_repeat(elem_size: int) -> int:
    if elem_size <= 2:
        return 128
    if elem_size <= 4:
        return 64
    return 32


def _compute_one_dim_plan(total_elements: int, elem_size: int, alive_num: int, ub_capacity_bytes: int) -> tuple[int, int, int, int, int, int]:
    cache_line = 128
    ub_former_byte = max(cache_line, (ub_capacity_bytes // max(alive_num, 1)))
    ub_former = max(1, (ub_former_byte // cache_line) * cache_line // max(elem_size, 1))
    ub_former = max(ALIGNMENT_BYTES // max(elem_size, 1), ub_former)
    ub_former = min(ub_former, MAX_REPEAT_TIMES * _elements_per_repeat(elem_size), max(total_elements, 1))

    core_num = 32
    ub_outer = max(1, (total_elements + ub_former - 1) // ub_former)
    ub_tail = total_elements % ub_former or ub_former
    block_former = max(1, (ub_outer + core_num - 1) // core_num)
    block_num = max(1, (ub_outer + block_former - 1) // block_former)

    if block_num < max(1, core_num // 2) and ub_former * elem_size * alive_num > 8 * 1024:
        dim_per_core = max(1, total_elements * 2 // core_num)
        aligned_dim_per_core = max(1, ceil_align(dim_per_core * elem_size, cache_line) // max(elem_size, 1))
        lowest_ub_former = max(1, ((8 * 1024 // max(alive_num, 1) // cache_line) * cache_line) // max(elem_size, 1))
        ub_former = min(ub_former, aligned_dim_per_core)
        ub_former = max(ub_former, lowest_ub_former)
        ub_former = min(ub_former, MAX_REPEAT_TIMES * _elements_per_repeat(elem_size), max(total_elements, 1))
        ub_outer = max(1, (total_elements + ub_former - 1) // ub_former)
        ub_tail = total_elements % ub_former or ub_former
        block_former = max(1, (ub_outer + core_num - 1) // core_num)
        block_num = max(1, (ub_outer + block_former - 1) // block_former)

    block_tail = ub_outer - (block_num - 1) * block_former
    return ub_former, ub_outer, ub_tail, block_former, block_num, block_tail


def _compute_multidim_plan(
    output_shape: List[int],
    elem_size: int,
    buffer_num: int,
    ub_capacity_bytes: int,
    extra_size: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    usable_bytes = max(ALIGNMENT_BYTES, ub_capacity_bytes - max(extra_size, 0))
    align_elems = max(1, ALIGNMENT_BYTES // max(elem_size, 1))
    max_elem_num = usable_bytes // max(buffer_num * elem_size, 1)
    max_elem_num = (max_elem_num // align_elems) * align_elems
    max_elem_num = min(max_elem_num, MAX_REPEAT_TIMES * _elements_per_repeat(elem_size))
    max_elem_num = max(align_elems, max_elem_num)

    cur_product = 1
    ub_split_axis = 0
    all_fit = True
    for axis in range(len(output_shape) - 1, -1, -1):
        next_product = cur_product * output_shape[axis]
        if next_product > max_elem_num:
            ub_split_axis = axis
            all_fit = False
            break
        cur_product = next_product

    if all_fit:
        if len(output_shape) > 1:
            cur_product = max(1, cur_product // output_shape[0])
        else:
            cur_product = max_elem_num

    ub_former = max(1, max_elem_num // max(cur_product, 1))
    ub_former = min(ub_former, output_shape[ub_split_axis])
    ub_outer = max(1, (output_shape[ub_split_axis] + ub_former - 1) // ub_former)
    ub_tail = output_shape[ub_split_axis] - (ub_outer - 1) * ub_former
    fused_product = ub_outer
    for axis in range(ub_split_axis):
        fused_product *= output_shape[axis]

    core_num = 32
    block_former = max(1, (fused_product + core_num - 1) // core_num)
    block_num = max(1, (fused_product + block_former - 1) // block_former)
    block_tail = fused_product - (block_num - 1) * block_former
    tile_length = ub_former * max(cur_product, 1)
    axes_inside_ub = len(output_shape) - ub_split_axis
    return ub_split_axis, ub_former, ub_outer, ub_tail, block_former, block_num, block_tail, tile_length, axes_inside_ub


def compute_broadcast_tiling(
    *,
    total_elements: int,
    dtype: str,
    intermediate_buffers: int = 0,
    ub_capacity_bytes: int = DEFAULT_UB_CAPACITY,
    op_name: str = "",
    query: str = "",
    input_shapes: Any = None,
    input_shape: Any = None,
    output_shape: Any = None,
    chip: Any = "DAV_2201",
) -> Optional[TilingParamsResult]:
    normalized_input_shapes = _normalize_input_shapes(input_shapes, input_shape)
    normalized_output_shape = normalize_shape(output_shape)
    if not normalized_input_shapes and not normalized_output_shape:
        return None

    normalized_output_shape = _infer_output_shape(normalized_input_shapes, normalized_output_shape)
    if not normalized_input_shapes or not normalized_output_shape:
        return build_unsupported_tiling_result(
            operator_class="broadcast",
            strategy_kind="broadcast_shape_mismatch",
            reason="broadcast tiling needs broadcast-compatible input/output shapes after left-padding dimensions",
            required_inputs=["input shapes", "output shape"],
        )

    collapsed_inputs, collapsed_output, input_strides, output_strides, collapsed_flags = _collapse_broadcast_shapes(
        normalized_input_shapes,
        normalized_output_shape,
    )
    output_elements = num_elements(collapsed_output)
    elem_size = dtype_bytes(dtype)
    chip_name = _normalize_chip(chip)
    broadcast_axes = {
        index: [axis for axis in range(len(collapsed_output)) if collapsed_inputs[index][axis] == 1 and collapsed_output[axis] > 1]
        for index in range(len(collapsed_inputs))
    }
    broadcast_input_count = sum(1 for axes in broadcast_axes.values() if axes)
    stage_summaries = [
        f"chip={chip_name}",
        f"collapsed_output={collapsed_output}",
        f"output_strides={output_strides}",
        f"collapsed_flags={collapsed_flags}",
    ]
    for index, input_dims in enumerate(collapsed_inputs):
        stage_summaries.append(f"input{index}_shape={input_dims}")
        stage_summaries.append(f"input{index}_strides={input_strides[index]}")
        stage_summaries.append(f"input{index}_broadcast_axes={broadcast_axes[index]}")

    alive_num = max(2, len(collapsed_inputs) + 1 + intermediate_buffers)
    warnings: List[str] = []

    if len(collapsed_output) == 1:
        scalar_inputs = [index for index, dims in enumerate(collapsed_inputs) if dims[0] == 1 and collapsed_output[0] > 1]
        ub_former, ub_outer, ub_tail, block_former, block_num, block_tail = _compute_one_dim_plan(
            output_elements,
            elem_size,
            alive_num,
            ub_capacity_bytes,
        )
        ub_usage_bytes = ub_former * elem_size * alive_num
        strategy_kind = "broadcast_onedim_scalar" if scalar_inputs else "broadcast_onedim_contiguous"
        stage_summaries.extend(
            [
                f"scalar_inputs={scalar_inputs}",
                f"ub_outer={ub_outer}",
                f"ub_tail={ub_tail}",
                f"block_former={block_former}",
            ]
        )
        return TilingParamsResult(
            status=NUMERIC_OK_STATUS,
            supported=True,
            operator_class="broadcast",
            strategy_kind=strategy_kind,
            reason="",
            block_num=block_num,
            num_per_core=block_former,
            tail_num_last_core=block_tail,
            tile_length=ub_former,
            repeat_times=max(1, (ub_former + _elements_per_repeat(elem_size) - 1) // _elements_per_repeat(elem_size)),
            ub_usage_bytes=ub_usage_bytes,
            ub_usage_pct=round((ub_usage_bytes / ub_capacity_bytes) * 100.0, 1),
            formula_used=(
                "Broadcast OneDim tiling: DimensionCollapse -> 1D contiguous split with 128B alignment; "
                "scalar inputs use TensorScalar fast path when available."
            ),
            constraints_met=ub_usage_bytes <= ub_capacity_bytes,
            load_mode="onedim",
            output_elements=output_elements,
            normalized_shape=collapsed_output,
            stage_summaries=stage_summaries,
            warnings=warnings,
        )

    base_buffer_num = max(2, len(collapsed_inputs) + 1 + intermediate_buffers)
    route = "ub_static"
    extra_size = 0

    if chip_name == "DAV_3510":
        tail_bytes = collapsed_output[-1] * elem_size
        tail_aligned_32 = tail_bytes % 32 == 0
        has_nlast = any(
            any(axis < len(collapsed_output) - 1 for axis in axes) and (len(collapsed_output) - 1) not in axes
            for axes in broadcast_axes.values()
        )
        low_precision = normalize_operator_text(dtype) in {"int8", "uint8", "half", "float16", "bf16", "bfloat16"}
        if len(collapsed_output) <= 9 and ((has_nlast and tail_bytes >= 4096) or (low_precision and tail_aligned_32)):
            route = "dynamic_ub"
        else:
            route = "nddma"
    elif len(collapsed_output) > 2:
        route = "ub_static_multidim_metadata"
        warnings.append(
            "Collapsed broadcast rank > 2 on DAV_2201 is planned with metadata only; kernel generation still needs looped static broadcast expansion."
        )
    else:
        for input_index, axes in broadcast_axes.items():
            if axes == [1]:
                aligned = collapsed_inputs[input_index][0] * elem_size % 32 == 0
                if not aligned:
                    route = "transport_axis_last_dummy_fill" if collapsed_output[0] > 2 else "transport_axis_last_duplicate"
                    break
            if axes == [0]:
                aligned = collapsed_inputs[input_index][1] * elem_size % 32 == 0
                if not aligned:
                    route = "transport_axis_penultimate_copy"
                    break

    buffer_num = base_buffer_num + (broadcast_input_count if route in {"ub_static", "ub_static_multidim_metadata", "dynamic_ub"} else 0)
    if route in {"ub_static", "ub_static_multidim_metadata"}:
        extra_size = 4096 * max(1, broadcast_input_count)

    ub_split_axis, ub_former, ub_outer, ub_tail, block_former, block_num, block_tail, tile_length, axes_inside_ub = _compute_multidim_plan(
        collapsed_output,
        elem_size,
        buffer_num,
        ub_capacity_bytes,
        extra_size,
    )

    if chip_name == "DAV_3510" and route == "nddma":
        route = "nddma_without_loop" if axes_inside_ub <= 5 else "nddma_with_loop"

    if route.startswith("transport_axis_last"):
        load_mode = "transport_dummy_fill" if route.endswith("dummy_fill") else "transport_duplicate"
    elif route == "transport_axis_penultimate_copy":
        load_mode = "transport_copy"
    elif route.startswith("nddma"):
        load_mode = route
    elif route == "dynamic_ub":
        load_mode = "dynamic_ub"
    else:
        load_mode = "ub_static"

    ub_usage_bytes = tile_length * elem_size * buffer_num + extra_size
    repeat_times = max(1, (tile_length + _elements_per_repeat(elem_size) - 1) // _elements_per_repeat(elem_size))
    strategy_kind = f"broadcast_{route}"
    stage_summaries.extend(
        [
            f"ub_split_axis={ub_split_axis}",
            f"ub_former={ub_former}",
            f"ub_outer={ub_outer}",
            f"ub_tail={ub_tail}",
            f"axes_inside_ub={axes_inside_ub}",
            f"buffer_num={buffer_num}",
            f"extra_size={extra_size}",
        ]
    )
    if route == "dynamic_ub":
        warnings.append("Dynamic UB Broadcast is planned only for DAV_3510 and assumes srcInnerPad=false.")
    if route.startswith("nddma"):
        warnings.append("NDDMA route models GM-to-UB broadcast only; UB-internal broadcast still needs a separate kernel expansion path.")

    planner_only = route in {"dynamic_ub", "ub_static_multidim_metadata", "nddma_without_loop", "nddma_with_loop"}
    if planner_only:
        warnings.append(
            "planner_ok means the tool selected a broadcast route and tile shape, but the corresponding kernel expansion path is still operator-specific"
        )

    return TilingParamsResult(
        status=PLANNER_OK_STATUS if planner_only else NUMERIC_OK_STATUS,
        supported=True,
        operator_class="broadcast",
        strategy_kind=strategy_kind,
        reason="",
        block_num=block_num,
        num_per_core=block_former,
        tail_num_last_core=block_tail,
        tile_length=tile_length,
        repeat_times=repeat_times,
        ub_usage_bytes=ub_usage_bytes,
        ub_usage_pct=round((ub_usage_bytes / ub_capacity_bytes) * 100.0, 1),
        formula_used=(
            "Broadcast tiling: DimensionCollapse -> multidimensional UB split -> route selection among OneDim, UB static, transport fallback, dynamic UB, or NDDMA based on chip and broadcast axes."
        ),
        constraints_met=ub_usage_bytes <= ub_capacity_bytes,
        load_mode=load_mode,
        output_elements=output_elements,
        normalized_shape=collapsed_output,
        stage_summaries=stage_summaries,
        warnings=warnings,
    )