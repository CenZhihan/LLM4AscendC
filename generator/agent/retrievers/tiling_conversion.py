from typing import Any

from .tiling_classification import is_supported_transpose_permutation, is_transpose_keyword
from .tiling_common import dtype_bytes, normalize_permutation, normalize_shape, num_elements
from .tiling_constants import ALIGNMENT_BYTES
from .tiling_generic import compute_tiling_params
from .tiling_types import NUMERIC_OK_STATUS, TilingParamsResult
from .tiling_unsupported import build_category_unsupported_result, build_unsupported_tiling_result


def compute_conversion_tiling(
    *,
    total_elements: int,
    dtype: str,
    intermediate_buffers: int,
    ub_capacity_bytes: int,
    op_name: str,
    query: str,
    input_shape: Any,
    output_shape: Any,
    permutation: Any,
) -> TilingParamsResult:
    if not is_transpose_keyword(op_name, query) and not permutation:
        return build_category_unsupported_result("conversion", op_name=op_name, query=query)

    normalized_input_shape = normalize_shape(input_shape)
    normalized_output_shape = normalize_shape(output_shape)
    normalized_permutation = normalize_permutation(permutation)

    if not normalized_input_shape or not normalized_permutation:
        return build_unsupported_tiling_result(
            operator_class="conversion",
            strategy_kind="conversion_transpose_layout_required",
            reason="transpose partial support needs structured input_shape and permutation",
            required_inputs=["input_shape", "permutation", "buffer plan"],
        )

    rank = len(normalized_input_shape)
    if len(normalized_permutation) != rank or sorted(normalized_permutation) != list(range(rank)):
        return build_unsupported_tiling_result(
            operator_class="conversion",
            strategy_kind="conversion_permutation_required",
            reason="permutation must be a valid reordering of input dimensions",
            required_inputs=["input_shape", "permutation", "buffer plan"],
        )

    inferred_output_shape = [normalized_input_shape[index] for index in normalized_permutation]
    if normalized_output_shape and normalized_output_shape != inferred_output_shape:
        return build_unsupported_tiling_result(
            operator_class="conversion",
            strategy_kind="conversion_output_shape_mismatch",
            reason="output_shape does not match the permutation applied to input_shape",
            required_inputs=["input_shape", "output_shape", "permutation"],
        )

    if not is_supported_transpose_permutation(normalized_permutation):
        return build_unsupported_tiling_result(
            operator_class="conversion",
            strategy_kind="conversion_transpose_permutation_unsupported",
            reason="generic conversion support is limited to 2D transpose or swapping the last two dimensions",
            required_inputs=["supported permutation", "input_shape", "buffer plan"],
        )

    effective_total_elements = total_elements or num_elements(normalized_input_shape)
    conversion_buffers = max(intermediate_buffers + 1, 1)
    result = compute_tiling_params(
        total_elements=effective_total_elements,
        dtype=dtype,
        op_type="broadcast",
        intermediate_buffers=conversion_buffers,
        ub_capacity_bytes=ub_capacity_bytes,
    )
    result.status = NUMERIC_OK_STATUS
    result.operator_class = "conversion"
    result.strategy_kind = "conversion_transpose_last_two_dims"
    result.formula_used = (
        "Conversion tiling: transpose with contiguous tiles over total elements, "
        f"input_shape={normalized_input_shape}, permutation={normalized_permutation}. "
        + (result.formula_used or "")
    )
    if normalized_input_shape[-1] * dtype_bytes(dtype) % ALIGNMENT_BYTES != 0:
        result.warnings.append("innermost dimension is not 256B aligned; implementation may need tail handling")
    return result