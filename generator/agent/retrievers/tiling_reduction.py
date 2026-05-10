from typing import Any, Dict, List

from .tiling_common import (
    candidate_reduction_tile_lengths,
    ceil_align,
    compute_core_split,
    compute_reduce_tmpbuf_size,
    dtype_bytes,
    normalize_operator_text,
    normalize_reduction_axes,
    normalize_shape,
    num_elements,
)
from .tiling_constants import ALIGNMENT_BYTES, MAX_REPEAT_TIMES
from .tiling_types import NUMERIC_OK_STATUS, PLANNER_OK_STATUS, TilingParamsResult
from .tiling_unsupported import build_unsupported_tiling_result


def _find_ar_chunk_plan(r_length: int, elem_size: int, ub_capacity_bytes: int) -> tuple[int, int, int, int] | None:
    alignment_elems = max(1, 32 // max(elem_size, 1))
    per_repeat = max(1, ALIGNMENT_BYTES // max(elem_size, 1))
    max_chunk = min(r_length, MAX_REPEAT_TIMES * per_repeat)
    candidate = max_chunk

    while candidate > 0:
        aligned_chunk = ((candidate * elem_size + 31) // 32) * 32 // elem_size
        tmp_buf_size = compute_reduce_tmpbuf_size(aligned_chunk, elem_size)
        ub_usage_bytes = aligned_chunk * elem_size + 64 + tmp_buf_size
        repeat_times = max(1, (aligned_chunk + per_repeat - 1) // per_repeat)
        if ub_usage_bytes <= ub_capacity_bytes and repeat_times <= MAX_REPEAT_TIMES:
            return candidate, aligned_chunk, ub_usage_bytes, repeat_times
        if candidate <= alignment_elems:
            candidate -= 1
        else:
            candidate -= alignment_elems

    return None


def _find_ara_full_load_plan(
    r_length: int,
    a0: int,
    elem_size: int,
    ub_capacity_bytes: int,
) -> tuple[int, int, int] | None:
    if r_length > MAX_REPEAT_TIMES:
        return None

    for tile_a0_len in candidate_reduction_tile_lengths(a0, elem_size):
        aligned_cols = ((tile_a0_len * elem_size + 31) // 32) * 32 // elem_size
        tmp_buf_size = compute_reduce_tmpbuf_size(r_length * aligned_cols, elem_size)
        ub_usage_bytes = 2 * r_length * aligned_cols * elem_size + 2 * aligned_cols * elem_size + tmp_buf_size
        if ub_usage_bytes <= ub_capacity_bytes:
            return tile_a0_len, aligned_cols, ub_usage_bytes
    return None


def _find_ara_rowsplit_plan(
    r_length: int,
    a0: int,
    elem_size: int,
    ub_capacity_bytes: int,
) -> tuple[int, int, int, int] | None:
    for tile_a0_len in candidate_reduction_tile_lengths(a0, elem_size):
        aligned_cols = ((tile_a0_len * elem_size + 31) // 32) * 32 // elem_size
        for r_chunk_size in range(min(r_length, MAX_REPEAT_TIMES), 0, -1):
            tmp_buf_size = compute_reduce_tmpbuf_size(r_chunk_size * aligned_cols, elem_size)
            ub_usage_bytes = r_chunk_size * aligned_cols * elem_size + 3 * aligned_cols * elem_size + tmp_buf_size
            if ub_usage_bytes <= ub_capacity_bytes:
                return tile_a0_len, aligned_cols, r_chunk_size, ub_usage_bytes
    return None


def compute_reduction_tiling(
    *,
    total_elements: int,
    dtype: str,
    ub_capacity_bytes: int,
    input_shape: Any,
    reduction_axes: Any,
    keepdim: Any,
    track_index: Any,
    op_name: str,
    query: str,
    output_count: Any = None,
    algorithm_hint: Any = None,
    precision_sensitive: Any = None,
) -> TilingParamsResult:
    normalized_input_shape = normalize_shape(input_shape)
    if not normalized_input_shape:
        return build_unsupported_tiling_result(
            operator_class="reduction",
            strategy_kind="reduction_axis_required",
            reason="reduction partial support needs structured input_shape and reduction_axes",
            required_inputs=["input_shape", "reduction_axes", "keepdim flag"],
        )

    axes = normalize_reduction_axes(reduction_axes, len(normalized_input_shape))
    if not axes:
        return build_unsupported_tiling_result(
            operator_class="reduction",
            strategy_kind="reduction_axis_required",
            reason="reduction partial support needs a valid reduction_axes specification",
            required_inputs=["input_shape", "reduction_axes", "keepdim flag"],
        )

    reduction_text = normalize_operator_text(" ".join(filter(None, [op_name, query])))
    effective_total_elements = total_elements or num_elements(normalized_input_shape)
    keepdim_flag = bool(keepdim)
    track_index_flag = bool(track_index) or any(token in reduction_text for token in ["argmax", "argmin", "with_index"])

    try:
        requested_output_count = int(output_count) if output_count not in (None, "") else 0
    except (TypeError, ValueError):
        requested_output_count = 0

    inferred_output_count = requested_output_count or (
        2
        if track_index_flag or any(
            token in reduction_text
            for token in ["variance", "reduce_var", "reduce_std", "std", "bn_training_reduce", "square_sum", "std_with_mean"]
        )
        else 1
    )

    explicit_algorithm_hint = normalize_operator_text(algorithm_hint)
    precision_sensitive_flag = bool(precision_sensitive) or "dichotomy" in reduction_text or "precision_sensitive" in reduction_text
    elem_size = dtype_bytes(dtype)
    compare_align_elems = ALIGNMENT_BYTES // 4

    def _collapse_segments() -> List[tuple[str, int]]:
        segments: List[tuple[str, int]] = []
        axis_set = set(axes)
        for index, dim in enumerate(normalized_input_shape):
            if dim == 1:
                continue
            kind = "R" if index in axis_set else "A"
            if segments and segments[-1][0] == kind:
                segments[-1] = (kind, segments[-1][1] * dim)
            else:
                segments.append((kind, dim))
        if not segments:
            segments = [("R", 1)]
        if segments[0][0] == "R":
            segments.insert(0, ("A", 1))
        return segments

    def _stage_summary(stage: Dict[str, Any]) -> str:
        parts = [
            stage["strategy_kind"],
            f"A1={stage['a1']}",
            f"R={stage['r_length']}",
            f"A0={stage['a0']}",
            f"tile_length={stage['tile_length']}",
        ]
        if stage.get("chunk_count"):
            parts.append(f"chunk_count={stage['chunk_count']}")
        if stage.get("chunk_size"):
            parts.append(f"chunk_size={stage['chunk_size']}")
        return ", ".join(parts)

    def _plan_stage(a1: int, r_length: int, a0: int) -> Dict[str, Any] | None:
        if a0 <= 1:
            r_length_align = ceil_align(r_length * elem_size, 32) // elem_size
            tmp_buf_size = compute_reduce_tmpbuf_size(r_length_align, max(elem_size, 4 if track_index_flag else elem_size))
            per_repeat = ALIGNMENT_BYTES // max(elem_size, 1)
            repeat_times = max(1, (r_length_align + per_repeat - 1) // per_repeat)
            block_num, rows_per_core, tail_rows = compute_core_split(a1)

            if track_index_flag:
                full_load_ub = 2 * r_length_align * elem_size + 2 * 4 + max(tmp_buf_size * 2, 4096)
                if full_load_ub <= ub_capacity_bytes and repeat_times <= MAX_REPEAT_TIMES:
                    return {
                        "strategy_kind": "reduction_with_index_ar_full_load",
                        "load_mode": "full_load",
                        "block_num": block_num,
                        "num_per_core": rows_per_core * r_length,
                        "tail_num_last_core": tail_rows * r_length,
                        "tile_length": r_length_align,
                        "repeat_times": repeat_times,
                        "ub_usage_bytes": full_load_ub,
                        "warnings": ["index is stored as float before cast to int32"],
                        "a1": a1,
                        "r_length": r_length,
                        "a0": 1,
                        "output_elements": 1,
                    }

                ar_chunk_plan = _find_ar_chunk_plan(r_length, elem_size, ub_capacity_bytes)
                if ar_chunk_plan is None:
                    return None
                chunk_cols, aligned_chunk_cols, _, chunk_repeat_times = ar_chunk_plan
                chunk_tmp_buf = compute_reduce_tmpbuf_size(aligned_chunk_cols, 4)
                chunk_ub_usage = aligned_chunk_cols * elem_size + 64 + chunk_tmp_buf
                chunk_count = (r_length + chunk_cols - 1) // chunk_cols
                last_chunk_size = r_length - (chunk_count - 1) * chunk_cols
                return {
                    "strategy_kind": "reduction_with_index_ar_col_split",
                    "load_mode": "col_split",
                    "block_num": block_num,
                    "num_per_core": rows_per_core * r_length,
                    "tail_num_last_core": tail_rows * r_length,
                    "tile_length": aligned_chunk_cols,
                    "repeat_times": chunk_repeat_times,
                    "ub_usage_bytes": chunk_ub_usage,
                    "warnings": [
                        "index is tracked per chunk and merged with chunk offsets",
                        "last chunk may require DataCopyPad for 32B alignment",
                    ],
                    "chunk_size": chunk_cols,
                    "chunk_count": chunk_count,
                    "last_chunk_size": last_chunk_size,
                    "a1": a1,
                    "r_length": r_length,
                    "a0": 1,
                    "output_elements": 1,
                }

            full_load_ub = 2 * r_length_align * elem_size + 64 + tmp_buf_size
            if full_load_ub <= ub_capacity_bytes and repeat_times <= MAX_REPEAT_TIMES:
                return {
                    "strategy_kind": "reduction_ar_full_load",
                    "load_mode": "full_load",
                    "block_num": block_num,
                    "num_per_core": rows_per_core * r_length,
                    "tail_num_last_core": tail_rows * r_length,
                    "tile_length": r_length_align,
                    "repeat_times": repeat_times,
                    "ub_usage_bytes": full_load_ub,
                    "warnings": [],
                    "a1": a1,
                    "r_length": r_length,
                    "a0": 1,
                    "output_elements": 1,
                }

            ar_chunk_plan = _find_ar_chunk_plan(r_length, elem_size, ub_capacity_bytes)
            if ar_chunk_plan is None:
                return None
            chunk_cols, aligned_chunk_cols, chunk_ub_usage, chunk_repeat_times = ar_chunk_plan
            chunk_count = (r_length + chunk_cols - 1) // chunk_cols
            last_chunk_size = r_length - (chunk_count - 1) * chunk_cols
            return {
                "strategy_kind": "reduction_ar_col_split",
                "load_mode": "col_split",
                "block_num": block_num,
                "num_per_core": rows_per_core * r_length,
                "tail_num_last_core": tail_rows * r_length,
                "tile_length": aligned_chunk_cols,
                "repeat_times": chunk_repeat_times,
                "ub_usage_bytes": chunk_ub_usage,
                "warnings": ["last chunk may require DataCopyPad for 32B alignment"] if chunk_cols != aligned_chunk_cols else [],
                "chunk_size": chunk_cols,
                "chunk_count": chunk_count,
                "last_chunk_size": last_chunk_size,
                "a1": a1,
                "r_length": r_length,
                "a0": 1,
                "output_elements": 1,
            }

        if track_index_flag:
            for tile_a0_len in candidate_reduction_tile_lengths(a0, 4):
                aligned_cols = ceil_align(tile_a0_len, compare_align_elems)
                cmp_buf_size = max(aligned_cols // 8, 32)
                full_load_ub = 2 * r_length * aligned_cols * elem_size + 4 * aligned_cols * 4 + cmp_buf_size
                repeat_times = max(1, (aligned_cols + compare_align_elems - 1) // compare_align_elems)
                if full_load_ub <= ub_capacity_bytes and repeat_times <= MAX_REPEAT_TIMES:
                    a0_outer = (a0 + tile_a0_len - 1) // tile_a0_len
                    total_tiles = a1 * a0_outer
                    block_num, tiles_per_core, tail_tiles = compute_core_split(total_tiles)
                    return {
                        "strategy_kind": "reduction_with_index_ara_full_load",
                        "load_mode": "full_load",
                        "block_num": block_num,
                        "num_per_core": tiles_per_core * r_length * tile_a0_len,
                        "tail_num_last_core": tail_tiles * r_length * tile_a0_len,
                        "tile_length": aligned_cols,
                        "repeat_times": repeat_times,
                        "ub_usage_bytes": full_load_ub,
                        "warnings": [
                            "index is stored as float before cast to int32",
                            "Compare/Select requires 256B alignment on the kept dimension",
                        ],
                        "tile_a0_len": tile_a0_len,
                        "aligned_cols": aligned_cols,
                        "a1": a1,
                        "r_length": r_length,
                        "a0": a0,
                        "output_elements": tile_a0_len,
                    }

            for tile_a0_len in candidate_reduction_tile_lengths(a0, 4):
                aligned_cols = ceil_align(tile_a0_len, compare_align_elems)
                cmp_buf_size = max(aligned_cols // 8, 32)
                for r_chunk_size in range(min(r_length, MAX_REPEAT_TIMES), 0, -1):
                    row_split_ub = r_chunk_size * aligned_cols * elem_size + 4 * aligned_cols * 4 + cmp_buf_size
                    if row_split_ub <= ub_capacity_bytes:
                        a0_outer = (a0 + tile_a0_len - 1) // tile_a0_len
                        total_tiles = a1 * a0_outer
                        block_num, tiles_per_core, tail_tiles = compute_core_split(total_tiles)
                        r_chunks = (r_length + r_chunk_size - 1) // r_chunk_size
                        r_last_chunk_size = r_length - (r_chunks - 1) * r_chunk_size
                        return {
                            "strategy_kind": "reduction_with_index_ara_row_split",
                            "load_mode": "row_split",
                            "block_num": block_num,
                            "num_per_core": tiles_per_core * r_length * tile_a0_len,
                            "tail_num_last_core": tail_tiles * r_length * tile_a0_len,
                            "tile_length": aligned_cols,
                            "repeat_times": max(1, (aligned_cols + compare_align_elems - 1) // compare_align_elems),
                            "ub_usage_bytes": row_split_ub,
                            "warnings": [
                                "index is stored as float before cast to int32",
                                "Compare/Select requires 256B alignment on the kept dimension",
                            ],
                            "chunk_size": r_chunk_size,
                            "chunk_count": r_chunks,
                            "last_chunk_size": r_last_chunk_size,
                            "tile_a0_len": tile_a0_len,
                            "aligned_cols": aligned_cols,
                            "a1": a1,
                            "r_length": r_length,
                            "a0": a0,
                            "output_elements": tile_a0_len,
                        }
            return None

        full_load_plan = _find_ara_full_load_plan(r_length, a0, elem_size, ub_capacity_bytes)
        if full_load_plan is not None:
            tile_a0_len, aligned_cols, ub_usage_bytes = full_load_plan
            a0_outer = (a0 + tile_a0_len - 1) // tile_a0_len
            total_tiles = a1 * a0_outer
            block_num, tiles_per_core, tail_tiles = compute_core_split(total_tiles)
            warnings: List[str] = []
            if aligned_cols != tile_a0_len:
                warnings.append("A0 tile is padded to 32B alignment for RA reduction")
            return {
                "strategy_kind": "reduction_ara_full_load",
                "load_mode": "full_load",
                "block_num": block_num,
                "num_per_core": tiles_per_core * r_length * tile_a0_len,
                "tail_num_last_core": tail_tiles * r_length * tile_a0_len,
                "tile_length": aligned_cols,
                "repeat_times": r_length,
                "ub_usage_bytes": ub_usage_bytes,
                "warnings": warnings,
                "tile_a0_len": tile_a0_len,
                "aligned_cols": aligned_cols,
                "a1": a1,
                "r_length": r_length,
                "a0": a0,
                "output_elements": tile_a0_len,
            }

        row_split_plan = _find_ara_rowsplit_plan(r_length, a0, elem_size, ub_capacity_bytes)
        if row_split_plan is None:
            return None
        tile_a0_len, aligned_cols, r_chunk_size, ub_usage_bytes = row_split_plan
        a0_outer = (a0 + tile_a0_len - 1) // tile_a0_len
        total_tiles = a1 * a0_outer
        block_num, tiles_per_core, tail_tiles = compute_core_split(total_tiles)
        r_chunks = (r_length + r_chunk_size - 1) // r_chunk_size
        r_last_chunk_size = r_length - (r_chunks - 1) * r_chunk_size
        warnings: List[str] = []
        if aligned_cols != tile_a0_len:
            warnings.append("A0 tile is padded to 32B alignment for RA reduction")
        return {
            "strategy_kind": "reduction_ara_row_split",
            "load_mode": "row_split",
            "block_num": block_num,
            "num_per_core": tiles_per_core * r_length * tile_a0_len,
            "tail_num_last_core": tail_tiles * r_length * tile_a0_len,
            "tile_length": aligned_cols,
            "repeat_times": r_chunk_size,
            "ub_usage_bytes": ub_usage_bytes,
            "warnings": warnings,
            "chunk_size": r_chunk_size,
            "chunk_count": r_chunks,
            "last_chunk_size": r_last_chunk_size,
            "tile_a0_len": tile_a0_len,
            "aligned_cols": aligned_cols,
            "a1": a1,
            "r_length": r_length,
            "a0": a0,
            "output_elements": tile_a0_len,
        }

    collapsed_segments = _collapse_segments()
    collapsed_pattern = "".join(kind for kind, _ in collapsed_segments)
    collapsed_shape = [size for _, size in collapsed_segments]
    reduced_segment_indices = [index for index, (kind, _) in enumerate(collapsed_segments) if kind == "R"]
    stage_plans: List[Dict[str, Any]] = []

    for segment_index in reduced_segment_indices:
        r_length = collapsed_segments[segment_index][1]
        a1 = 1
        for kind, size in collapsed_segments[:segment_index]:
            if kind == "A":
                a1 *= size
        a0 = 1
        for kind, size in collapsed_segments[segment_index + 1:]:
            if kind == "A":
                a0 *= size
        stage_plan = _plan_stage(a1, r_length, a0)
        if stage_plan is None:
            return build_unsupported_tiling_result(
                operator_class="reduction",
                strategy_kind="reduction_stage_plan_required",
                reason="one reduction stage still needs an operator-specific plan under current UB constraints",
                required_inputs=["input_shape", "reduction_axes", "stage-specific buffer plan"],
            )
        stage_plans.append(stage_plan)

    peak_stage = max(stage_plans, key=lambda item: item.get("ub_usage_bytes") or 0)
    is_multi_axis = len(stage_plans) > 1
    output_elements = max(1, num_elements([dim for index, dim in enumerate(normalized_input_shape) if index not in axes]))
    all_axes_reduced = len(axes) == len(normalized_input_shape)

    algorithm_kind = explicit_algorithm_hint
    if not algorithm_kind:
        if track_index_flag:
            algorithm_kind = "with_index"
        elif inferred_output_count >= 2:
            has_split_stage = any("split" in stage["strategy_kind"] for stage in stage_plans)
            algorithm_kind = "welford" if has_split_stage or is_multi_axis else "two_pass"
            if algorithm_kind == "welford" and any((stage.get("chunk_count") or 0) > 8 for stage in stage_plans):
                algorithm_kind = "welford_group8"
        elif "sum" in reduction_text and precision_sensitive_flag:
            algorithm_kind = "dichotomy"
        else:
            algorithm_kind = "direct"

    group_count = None
    workspace_bytes = None
    reduce_extent = max(stage["r_length"] for stage in stage_plans)
    should_group_reduce = algorithm_kind == "group_reduce" or (
        output_elements < min(4, (peak_stage.get("block_num") or 1))
        and reduce_extent > max(4096, (peak_stage.get("chunk_size") or peak_stage.get("tile_length") or 1) * 8)
    )
    if should_group_reduce:
        per_group_span = max(peak_stage.get("chunk_size") or peak_stage.get("repeat_times") or 1, 1)
        group_count = min(32, max(2, (reduce_extent + per_group_span - 1) // per_group_span))
        slot_bytes = ceil_align(max(1, output_elements) * max(1, inferred_output_count) * 4, ALIGNMENT_BYTES)
        workspace_bytes = group_count * slot_bytes
        if inferred_output_count >= 2:
            algorithm_kind = "group_welford"
        else:
            algorithm_kind = "group_reduce"

    strategy_kind = peak_stage["strategy_kind"]
    if is_multi_axis:
        strategy_kind = "reduction_multi_axis_nested"
    elif all_axes_reduced and strategy_kind.startswith("reduction_ar_"):
        strategy_kind = strategy_kind.replace("reduction_ar_", "reduction_global_", 1)
    elif all_axes_reduced and strategy_kind.startswith("reduction_with_index_ar_"):
        strategy_kind = strategy_kind.replace("reduction_with_index_ar_", "reduction_with_index_global_", 1)

    warnings = list(peak_stage.get("warnings") or [])
    if is_multi_axis:
        warnings.append("multi-axis reduction is modeled as nested single-axis stages after A/R collapse")
    if inferred_output_count >= 2 and algorithm_kind in {"two_pass", "welford", "welford_group8", "group_welford"}:
        warnings.append("multi-output reduction requires extra accumulator buffers beyond the scalar baseline")
    if algorithm_kind == "dichotomy":
        warnings.append("sum reduction is marked precision-sensitive; dichotomy accumulation is recommended")

    formula_parts = [
        f"input_shape={normalized_input_shape}",
        f"axes={axes}",
        f"collapsed_pattern={collapsed_pattern}",
        f"collapsed_shape={collapsed_shape}",
        f"algorithm={algorithm_kind}",
        f"keepdim={keepdim_flag}",
        f"total_elements={effective_total_elements}",
    ]
    if group_count is not None:
        formula_parts.append(f"group_count={group_count}")
        formula_parts.append(f"workspace_bytes={workspace_bytes}")
    formula_parts.append("stages=" + " | ".join(_stage_summary(stage) for stage in stage_plans))

    planner_only = (
        is_multi_axis
        or (inferred_output_count >= 2 and not track_index_flag)
        or algorithm_kind in {"welford", "welford_group8", "group_reduce", "group_welford", "dichotomy"}
    )
    if planner_only:
        warnings.append(
            "planner_ok means the tool resolved stage routing and sizing, but operator-specific kernel synthesis still needs to realize the algorithm details"
        )

    return TilingParamsResult(
        status=PLANNER_OK_STATUS if planner_only else NUMERIC_OK_STATUS,
        supported=True,
        operator_class="reduction",
        strategy_kind=strategy_kind,
        reason="",
        required_inputs=[],
        block_num=peak_stage.get("block_num"),
        num_per_core=peak_stage.get("num_per_core"),
        tail_num_last_core=peak_stage.get("tail_num_last_core"),
        tile_length=peak_stage.get("tile_length"),
        repeat_times=max(stage.get("repeat_times") or 0 for stage in stage_plans),
        ub_usage_bytes=peak_stage.get("ub_usage_bytes"),
        ub_usage_pct=round(((peak_stage.get("ub_usage_bytes") or 0) / ub_capacity_bytes) * 100.0, 1),
        formula_used="Reduction tiling: " + ", ".join(formula_parts),
        constraints_met=True,
        algorithm_kind=algorithm_kind,
        load_mode="nested" if is_multi_axis else peak_stage.get("load_mode"),
        output_count=inferred_output_count,
        output_elements=output_elements,
        workspace_bytes=workspace_bytes,
        group_count=group_count,
        chunk_size=peak_stage.get("chunk_size"),
        chunk_count=peak_stage.get("chunk_count"),
        last_chunk_size=peak_stage.get("last_chunk_size"),
        tile_a0_len=peak_stage.get("tile_a0_len"),
        aligned_cols=peak_stage.get("aligned_cols"),
        collapsed_pattern=collapsed_pattern,
        normalized_shape=collapsed_shape,
        normalized_axes=axes,
        stage_summaries=[_stage_summary(stage) for stage in stage_plans],
        warnings=warnings,
    )