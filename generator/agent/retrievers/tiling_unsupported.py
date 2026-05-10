from .tiling_classification import normalize_operator_class
from .tiling_common import normalize_operator_text
from .tiling_types import TilingParamsResult


def build_unsupported_tiling_result(
    *,
    operator_class: str,
    reason: str,
    required_inputs=None,
    strategy_kind: str = "operator_specific_required",
) -> TilingParamsResult:
    return TilingParamsResult(
        status="unsupported_without_operator_specific_strategy",
        supported=False,
        operator_class=operator_class or "unknown",
        strategy_kind=strategy_kind,
        reason=reason,
        required_inputs=list(required_inputs or []),
    )


def build_category_unsupported_result(
    operator_class: str,
    *,
    op_name: str = "",
    query: str = "",
) -> TilingParamsResult:
    text = " ".join(filter(None, [normalize_operator_text(op_name), normalize_operator_text(query)]))

    if operator_class == "reduction":
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="reduction_axis_required",
            reason="reduction tiling needs axis-aware multi-core split and reduction-specific work buffers",
            required_inputs=["reduction axis", "keepdim flag", "buffer plan"],
        )

    if operator_class == "conversion":
        if "transpose" in text or "permute" in text:
            return build_unsupported_tiling_result(
                operator_class=operator_class,
                strategy_kind="conversion_transpose_layout_required",
                reason="transpose tiling needs permutation- and layout-aware chunking",
                required_inputs=["input shape", "permutation", "layout plan", "buffer plan"],
            )
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="conversion_layout_required",
            reason="conversion tiling needs layout-aware chunking and operator-specific buffer planning",
            required_inputs=["input shape", "output shape", "layout plan", "buffer plan"],
        )

    if operator_class == "random":
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="random_state_required",
            reason="random tiling needs RNG state management and distribution-specific vectorization",
            required_inputs=["distribution type", "seed/state policy", "buffer plan"],
        )

    if operator_class == "matmul":
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="matrix_blocking_required",
            reason="matmul tiling needs matrix blocking and L0/L1 buffer planning",
            required_inputs=["matrix shapes", "blocking strategy", "L0/L1 buffer plan"],
        )

    if operator_class == "convolution":
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="convolution_window_required",
            reason="convolution tiling needs window traversal, padding policy, and feature-map blocking",
            required_inputs=["kernel shape", "stride/padding", "feature-map blocking"],
        )

    if operator_class == "nn":
        return build_unsupported_tiling_result(
            operator_class=operator_class,
            strategy_kind="nn_operator_specific_required",
            reason="NN tiling needs operator-specific multi-stage scheduling and buffer planning",
            required_inputs=["operator structure", "reduction/window plan", "buffer plan"],
        )

    normalized = normalize_operator_class(operator_class)
    return build_unsupported_tiling_result(
        operator_class=normalized,
        strategy_kind="classification_required",
        reason="operator class is unknown, so generic tiling is unsafe",
        required_inputs=["operator category", "buffer plan", "loop design"],
    )