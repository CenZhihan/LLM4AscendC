"""
Tiling Retriever for Ascend C kernel development agent.

Provides conservative tiling parameter computation and validation based on
hardware constraints (repeatTimes <= 255, 256B alignment, UB capacity).
"""
from typing import Any, Dict

from .tiling_classification import classify_operator_for_tiling, find_blacklist_entry
from .tiling_broadcast import compute_broadcast_tiling
from .tiling_constants import DEFAULT_UB_CAPACITY, GENERIC_TILING_CLASSES
from .tiling_conversion import compute_conversion_tiling
from .tiling_generic import compute_generic_tiling_for_class, compute_tiling_params
from .tiling_reduction import compute_reduction_tiling
from .tiling_types import PLANNER_OK_STATUS, TilingParamsResult, TilingValidationResult, VALIDATABLE_TILING_STATUSES
from .tiling_unsupported import build_category_unsupported_result, build_unsupported_tiling_result
from .tiling_validation import build_skipped_validation_result, validate_tiling_params


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
        ub_capacity_bytes: int = DEFAULT_UB_CAPACITY,
        op_name: str = "",
        state_category: str = "",
        query: str = "",
        input_shapes: Any = None,
        input_shape: Any = None,
        output_shape: Any = None,
        permutation: Any = None,
        reduction_axes: Any = None,
        keepdim: Any = None,
        track_index: Any = None,
        output_count: Any = None,
        algorithm_hint: Any = None,
        precision_sensitive: Any = None,
        chip: Any = "DAV_2201",
    ) -> TilingParamsResult:
        def _missing_size_result(target_class: str) -> TilingParamsResult:
            return build_unsupported_tiling_result(
                operator_class=target_class,
                strategy_kind=f"{target_class}_size_required",
                reason="generic tiling now requires explicit total_elements or shape information; query-only fallback no longer invents a default workload",
                required_inputs=["total_elements or shape", "dtype"],
            )

        operator_class = classify_operator_for_tiling(
            args={"op_type": op_type},
            state_category=state_category,
            state_op_name=op_name,
            query=query,
        )
        blacklist_entry = find_blacklist_entry(op_name, query)
        if blacklist_entry is not None:
            return build_unsupported_tiling_result(
                operator_class=blacklist_entry["operator_class"],
                reason=blacklist_entry["reason"],
                required_inputs=blacklist_entry["required_inputs"],
            )

        if operator_class == "broadcast":
            broadcast_result = compute_broadcast_tiling(
                total_elements=total_elements,
                dtype=dtype,
                intermediate_buffers=intermediate_buffers,
                ub_capacity_bytes=ub_capacity_bytes,
                op_name=op_name,
                query=query,
                input_shapes=input_shapes,
                input_shape=input_shape,
                output_shape=output_shape,
                chip=chip,
            )
            if broadcast_result is not None:
                return broadcast_result
            return build_unsupported_tiling_result(
                operator_class=operator_class,
                strategy_kind="broadcast_shape_required",
                reason="broadcast tiling requires structured input/output shape information; keyword-only broadcast requests are no longer accepted",
                required_inputs=["input shapes", "output shape", "chip"],
            )

        if operator_class == "reduction":
            return compute_reduction_tiling(
                total_elements=total_elements,
                dtype=dtype,
                ub_capacity_bytes=ub_capacity_bytes,
                input_shape=input_shape,
                reduction_axes=reduction_axes,
                keepdim=keepdim,
                track_index=track_index,
                op_name=op_name,
                query=query,
                output_count=output_count,
                algorithm_hint=algorithm_hint,
                precision_sensitive=precision_sensitive,
            )

        if operator_class == "conversion":
            return compute_conversion_tiling(
                total_elements=total_elements,
                dtype=dtype,
                intermediate_buffers=intermediate_buffers,
                ub_capacity_bytes=ub_capacity_bytes,
                op_name=op_name,
                query=query,
                input_shape=input_shape,
                output_shape=output_shape,
                permutation=permutation,
            )

        if operator_class not in GENERIC_TILING_CLASSES:
            return build_category_unsupported_result(
                operator_class,
                op_name=op_name,
                query=query,
            )

        if total_elements <= 0:
            return _missing_size_result(operator_class)

        return compute_generic_tiling_for_class(
            total_elements=total_elements,
            dtype=dtype,
            operator_class=operator_class,
            intermediate_buffers=intermediate_buffers,
            ub_capacity_bytes=ub_capacity_bytes,
        )

    def validate_tiling(
        self,
        tiling_params: Dict,
        chip: str = "DAV_2201",
    ) -> TilingValidationResult:
        status = str(tiling_params.get("status") or "").strip().lower()
        effective_chip = str(tiling_params.get("chip") or chip)
        if status == PLANNER_OK_STATUS:
            return build_skipped_validation_result(
                "upstream tiling_calc returned status=planner_ok, which is route/planning metadata rather than a fully numeric tiling candidate"
            )
        if status and status not in VALIDATABLE_TILING_STATUSES:
            return build_skipped_validation_result(
                f"upstream tiling_calc returned status={status}, no numeric tiling parameters to validate"
            )
        return validate_tiling_params(tiling_params, effective_chip)


__all__ = [
    "TilingRetriever",
    "TilingParamsResult",
    "TilingValidationResult",
    "classify_operator_for_tiling",
    "compute_tiling_params",
    "validate_tiling_params",
]