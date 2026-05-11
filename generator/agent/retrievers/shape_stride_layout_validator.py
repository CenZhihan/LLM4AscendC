from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tiling_common import ceil_align, dtype_bytes, normalize_shape, num_elements


_DIRECTION_ALIASES = {
    "gm_to_ub": "GM_TO_UB",
    "gm->ub": "GM_TO_UB",
    "gm2ub": "GM_TO_UB",
    "ub_to_gm": "UB_TO_GM",
    "ub->gm": "UB_TO_GM",
    "ub2gm": "UB_TO_GM",
    "ub_to_ub": "UB_TO_UB",
    "ub->ub": "UB_TO_UB",
    "ub2ub": "UB_TO_UB",
}

_ROW_COPY_KINDS = {"ROW_WISE", "PAD_COPY", "BLOCK_COPY"}
_DEFAULT_ALIGNMENT_BYTES = 32


@dataclass
class RuleViolation:
    rule_id: str
    severity: str
    message: str


@dataclass
class RepairSuggestion:
    type: str
    summary: str
    rationale: str
    expected_status_after_fix: str
    patch_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShapeStrideLayoutValidatorResult:
    status: str
    reason: str = ""
    shape_status: str = "SHAPE_MISSING"
    layout_class: str = "UNKNOWN_LAYOUT"
    contiguity_status: str = "CONTIGUITY_UNKNOWN"
    is_contiguous: bool = False
    requires_rebuild: bool = False
    stride_status: str = "STRIDE_MISSING_UNSAFE"
    movement_status: str = "MOVEMENT_UNKNOWN"
    hardware_status: str = "HARDWARE_UNKNOWN"
    copy_param_status: str = "copy_params_missing_context"
    normalized_request: Dict[str, Any] = field(default_factory=dict)
    inferred_layout_details: Dict[str, Any] = field(default_factory=dict)
    violated_rules: List[RuleViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[RepairSuggestion] = field(default_factory=list)
    confidence: str = "LOW"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_direction(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    return _DIRECTION_ALIASES.get(raw, raw.upper())


def _normalize_copy_kind(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_")
    return text or "AUTO"


def _parse_stride(raw_stride: Any, rank: int) -> Tuple[List[int], str]:
    if raw_stride is None:
        return [], "SHAPE_MISSING"
    if not isinstance(raw_stride, (list, tuple)):
        return [], "SHAPE_INVALID_RANK"
    if len(raw_stride) != rank:
        return [], "SHAPE_STRIDE_RANK_MISMATCH"

    stride: List[int] = []
    for item in raw_stride:
        parsed = _maybe_int(item)
        if parsed is None or parsed < 0:
            return [], "STRIDE_VALUE_INVALID"
        stride.append(parsed)
    return stride, "SHAPE_VALID"


def _compute_dense_stride(shape: List[int]) -> List[int]:
    if not shape:
        return []
    dense = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        dense[index] = dense[index + 1] * shape[index + 1]
    return dense


def _infer_spaces(direction: str, request: Dict[str, Any]) -> Tuple[str, str]:
    src_space = str(request.get("src_space") or "").strip().upper()
    dst_space = str(request.get("dst_space") or "").strip().upper()
    if src_space and dst_space:
        return src_space, dst_space
    mapping = {
        "GM_TO_UB": ("GM", "UB"),
        "UB_TO_GM": ("UB", "GM"),
        "UB_TO_UB": ("UB", "UB"),
    }
    return mapping.get(direction, (src_space or "UNKNOWN", dst_space or "UNKNOWN"))


def _is_row_regular(shape: List[int], stride: List[int]) -> bool:
    if not shape or not stride or len(shape) != len(stride):
        return False
    if len(shape) == 1:
        return stride[0] == 1
    if stride[-1] != 1:
        return False
    if stride[-2] < shape[-1]:
        return False
    for index in range(len(shape) - 3, -1, -1):
        if stride[index] != shape[index + 1] * stride[index + 1]:
            return False
    return True


def _is_affine_permutation(shape: List[int], stride: List[int]) -> bool:
    if len(shape) <= 1 or not stride or any(item <= 0 for item in stride):
        return False
    order = sorted(range(len(shape)), key=lambda idx: (-stride[idx], idx))
    permuted_shape = [shape[idx] for idx in order]
    expected_dense = _compute_dense_stride(permuted_shape)
    remapped = [0] * len(shape)
    for idx, expected in zip(order, expected_dense):
        remapped[idx] = expected
    return remapped == stride


def _derive_layout(shape: List[int], stride: List[int], dense_stride: List[int]) -> Tuple[str, str, bool]:
    if not shape or not stride:
        return "UNKNOWN_LAYOUT", "CONTIGUITY_UNKNOWN", False
    if any(item == 0 for item in stride):
        return "BROADCAST_LIKE", "NONCONTIGUOUS_IRREGULAR", False
    if stride == dense_stride:
        return "DENSE_CONTIGUOUS", "FULLY_CONTIGUOUS", True
    if _is_row_regular(shape, stride):
        return "ROW_REGULAR_NONCONTIGUOUS", "INNER_CONTIGUOUS_ONLY", False
    if _is_affine_permutation(shape, stride):
        return "TRANSPOSED_REGULAR", "NONCONTIGUOUS_REGULAR", False
    return "IRREGULAR", "NONCONTIGUOUS_IRREGULAR", False


def _validate_stride_field(
    raw_value: Any,
    *,
    expected_gap_bytes: Optional[int],
    space: str,
    field_name: str,
    violations: List[RuleViolation],
    suggestions: List[RepairSuggestion],
    warnings: List[str],
) -> Tuple[str, Optional[int], Optional[int]]:
    if raw_value in (None, ""):
        if expected_gap_bytes is None:
            return "STRIDE_VALID", None, None
        warnings.append(f"{field_name} was omitted; inferred expected gap from layout analysis")
        return "STRIDE_MISSING_INFERABLE", expected_gap_bytes, (expected_gap_bytes // 32) if expected_gap_bytes % 32 == 0 else None

    parsed = _maybe_int(raw_value)
    if parsed is None or parsed < 0:
        violations.append(
            RuleViolation(
                rule_id="STRIDE_004",
                severity="error",
                message=f"{field_name} must be a non-negative integer",
            )
        )
        return "STRIDE_VALUE_INVALID", None, None

    normalized_bytes = parsed if space == "GM" else parsed * 32
    normalized_blocks = parsed if space == "UB" else (parsed // 32 if parsed % 32 == 0 else None)
    if expected_gap_bytes is None:
        return "STRIDE_VALID", normalized_bytes, normalized_blocks
    if normalized_bytes == expected_gap_bytes:
        return "STRIDE_VALID", normalized_bytes, normalized_blocks

    if space == "UB" and parsed == expected_gap_bytes and expected_gap_bytes % 32 == 0:
        corrected = expected_gap_bytes // 32
        suggestions.append(
            RepairSuggestion(
                type="FIX_STRIDE_UNIT",
                summary=f"Interpret {field_name} as UB dataBlock units instead of bytes",
                rationale=f"The affine gap is correct in bytes, but UB-side {field_name} must be encoded in 32-byte blocks.",
                expected_status_after_fix="VALID",
                patch_fields={field_name: corrected},
            )
        )
        violations.append(
            RuleViolation(
                rule_id="STRIDE_002",
                severity="error",
                message=f"{field_name} looks byte-based, but UB stride fields must use 32-byte blocks",
            )
        )
        return "STRIDE_UNIT_MISMATCH", normalized_bytes, normalized_blocks

    if space == "GM" and parsed * 32 == expected_gap_bytes:
        corrected = expected_gap_bytes
        suggestions.append(
            RepairSuggestion(
                type="FIX_STRIDE_UNIT",
                summary=f"Interpret {field_name} as GM byte stride instead of dataBlock units",
                rationale=f"The affine gap matches only after converting 32-byte blocks back to bytes, which is illegal on the GM side.",
                expected_status_after_fix="VALID",
                patch_fields={field_name: corrected},
            )
        )
        violations.append(
            RuleViolation(
                rule_id="STRIDE_001",
                severity="error",
                message=f"{field_name} looks block-based, but GM stride fields must use bytes",
            )
        )
        return "STRIDE_UNIT_MISMATCH", normalized_bytes, normalized_blocks

    violations.append(
        RuleViolation(
            rule_id="STRIDE_003",
            severity="error",
            message=f"{field_name} does not match the expected affine gap for the requested copy pattern",
        )
    )
    return "STRIDE_VALUE_INVALID", normalized_bytes, normalized_blocks


def _pick_stride_status(*statuses: str) -> str:
    priority = {
        "STRIDE_VALUE_INVALID": 5,
        "STRIDE_NOT_AFFINE": 4,
        "STRIDE_UNIT_MISMATCH": 3,
        "STRIDE_MISSING_UNSAFE": 2,
        "STRIDE_MISSING_INFERABLE": 1,
        "STRIDE_VALID": 0,
    }
    best = "STRIDE_VALID"
    best_score = -1
    for status in statuses:
        score = priority.get(status, 0)
        if score > best_score:
            best = status
            best_score = score
    return best


def _copy_param_status(
    status: str,
    movement_status: str,
    stride_status: str,
    requires_rebuild: bool,
) -> str:
    if status == "INSUFFICIENT_CONTEXT":
        return "copy_params_missing_context"
    if status == "INVALID" or stride_status in {"STRIDE_VALUE_INVALID", "STRIDE_NOT_AFFINE"}:
        return "copy_params_invalid"
    if requires_rebuild:
        return "copy_params_require_rebuild"
    if movement_status == "MOVEMENT_REQUIRES_PAD_COPY":
        return "copy_params_require_pad_copy"
    if movement_status == "MOVEMENT_ROW_SPLIT_REQUIRED":
        return "copy_params_requires_row_split"
    if stride_status == "STRIDE_MISSING_INFERABLE":
        return "copy_params_inferred"
    return "copy_params_valid"


def _confidence(stride_status: str, status: str, movement_direction: str) -> str:
    if status == "INSUFFICIENT_CONTEXT" or movement_direction not in {"GM_TO_UB", "UB_TO_GM", "UB_TO_UB"}:
        return "LOW"
    if stride_status == "STRIDE_MISSING_INFERABLE":
        return "MEDIUM"
    return "HIGH"


def plan_shape_stride_layout_validation(request: Dict[str, Any]) -> ShapeStrideLayoutValidatorResult:
    request = dict(request or {})
    missing_inputs: List[str] = []
    violations: List[RuleViolation] = []
    suggestions: List[RepairSuggestion] = []
    warnings: List[str] = []

    raw_shape = request.get("tensor_shape") if "tensor_shape" in request else request.get("shape")
    raw_stride = request.get("tensor_stride") if "tensor_stride" in request else request.get("stride")
    raw_direction = request.get("movement_direction") if "movement_direction" in request else request.get("direction")
    direction = _clean_direction(raw_direction)
    if direction not in {"GM_TO_UB", "UB_TO_GM", "UB_TO_UB"}:
        missing_inputs.append("movement_direction")

    shape = normalize_shape(raw_shape)
    if raw_shape is None:
        shape_status = "SHAPE_MISSING"
        missing_inputs.append("tensor_shape")
    elif not shape:
        shape_status = "SHAPE_INVALID_DIM"
        violations.append(
            RuleViolation(
                rule_id="SHAPE_002",
                severity="error",
                message="tensor_shape must be a non-empty list of positive integers",
            )
        )
    else:
        shape_status = "SHAPE_VALID"

    rank = len(shape)
    stride, stride_parse_status = _parse_stride(raw_stride, rank)
    if raw_stride is None:
        if "tensor_stride" not in missing_inputs:
            missing_inputs.append("tensor_stride")
        if shape_status == "SHAPE_VALID":
            shape_status = "SHAPE_MISSING"
    elif stride_parse_status == "SHAPE_STRIDE_RANK_MISMATCH":
        shape_status = "SHAPE_STRIDE_RANK_MISMATCH"
        violations.append(
            RuleViolation(
                rule_id="SHAPE_004",
                severity="error",
                message="tensor_stride must have the same rank as tensor_shape",
            )
        )
    elif stride_parse_status == "STRIDE_VALUE_INVALID":
        shape_status = "SHAPE_VALID"

    element_bytes = _maybe_int(request.get("element_bytes"))
    if element_bytes is None or element_bytes <= 0:
        dtype = str(request.get("element_dtype") or request.get("dtype") or "").strip()
        if not dtype:
            missing_inputs.append("element_dtype or element_bytes")
            element_bytes = 0
        else:
            element_bytes = dtype_bytes(dtype)
    dtype_name = str(request.get("element_dtype") or request.get("dtype") or "")

    if missing_inputs and shape_status == "SHAPE_MISSING":
        suggestions.append(
            RepairSuggestion(
                type="PROVIDE_MISSING_FIELDS",
                summary="Provide the missing structured tensor metadata",
                rationale="The validator cannot derive a reliable legality verdict without the required tensor metadata.",
                expected_status_after_fix="VALID",
                patch_fields={"missing_inputs": missing_inputs},
            )
        )

    if missing_inputs and (shape_status == "SHAPE_MISSING" or element_bytes <= 0 or direction not in {"GM_TO_UB", "UB_TO_GM", "UB_TO_UB"}):
        return ShapeStrideLayoutValidatorResult(
            status="INSUFFICIENT_CONTEXT",
            reason="missing required structured inputs for shape/stride validation",
            shape_status=shape_status,
            normalized_request={
                "movement_direction": direction,
                "tensor_shape": raw_shape,
                "tensor_stride": raw_stride,
            },
            violated_rules=violations,
            warnings=warnings,
            suggestions=suggestions,
            confidence="LOW",
        )

    if shape_status != "SHAPE_VALID" or not stride or element_bytes <= 0:
        return ShapeStrideLayoutValidatorResult(
            status="INVALID",
            reason="shape or stride metadata is malformed",
            shape_status=shape_status,
            stride_status="STRIDE_VALUE_INVALID" if stride_parse_status == "STRIDE_VALUE_INVALID" else "STRIDE_MISSING_UNSAFE",
            normalized_request={
                "movement_direction": direction,
                "tensor_shape": shape,
                "tensor_stride": raw_stride,
            },
            violated_rules=violations,
            warnings=warnings,
            suggestions=suggestions,
            confidence="LOW",
        )

    dense_stride = _compute_dense_stride(shape)
    layout_class, contiguity_status, is_contiguous = _derive_layout(shape, stride, dense_stride)
    total_elements = num_elements(shape)
    src_space, dst_space = _infer_spaces(direction, request)
    requested_copy_kind = _normalize_copy_kind(request.get("requested_copy_kind"))
    gm_alignment_bytes = max(1, _safe_int(request.get("gm_alignment_bytes"), _DEFAULT_ALIGNMENT_BYTES))
    ub_alignment_bytes = max(1, _safe_int(request.get("ub_alignment_bytes"), _DEFAULT_ALIGNMENT_BYTES))

    use_row_view = (
        requested_copy_kind in _ROW_COPY_KINDS
        or _maybe_int(request.get("row_count")) not in (None, 0)
        or _maybe_int(request.get("row_bytes")) not in (None, 0)
        or layout_class == "ROW_REGULAR_NONCONTIGUOUS"
    )

    derived_row_count = _safe_int(request.get("row_count"), 0)
    derived_row_bytes = _safe_int(request.get("row_bytes"), 0)
    if use_row_view:
        if derived_row_count <= 0:
            derived_row_count = (total_elements // shape[-1]) if len(shape) > 1 else 1
        if derived_row_bytes <= 0:
            derived_row_bytes = shape[-1] * element_bytes
    else:
        if derived_row_count <= 0:
            derived_row_count = 1
        if derived_row_bytes <= 0:
            derived_row_bytes = total_elements * element_bytes

    expected_src_gap_bytes: Optional[int] = None
    if use_row_view and derived_row_count > 1 and layout_class == "ROW_REGULAR_NONCONTIGUOUS":
        expected_src_gap_bytes = (stride[-2] - shape[-1]) * element_bytes
    elif use_row_view and derived_row_count > 1 and layout_class == "DENSE_CONTIGUOUS":
        expected_src_gap_bytes = 0

    expected_dst_gap_bytes: Optional[int] = None
    if use_row_view and derived_row_count > 1:
        expected_dst_gap_bytes = 0
        if dst_space == "UB":
            expected_dst_gap_bytes = ceil_align(derived_row_bytes, ub_alignment_bytes) - derived_row_bytes

    src_stride_status, src_stride_bytes, src_stride_blocks = _validate_stride_field(
        request.get("src_stride"),
        expected_gap_bytes=expected_src_gap_bytes,
        space=src_space,
        field_name="src_stride",
        violations=violations,
        suggestions=suggestions,
        warnings=warnings,
    )
    dst_stride_status, dst_stride_bytes, dst_stride_blocks = _validate_stride_field(
        request.get("dst_stride"),
        expected_gap_bytes=expected_dst_gap_bytes,
        space=dst_space,
        field_name="dst_stride",
        violations=violations,
        suggestions=suggestions,
        warnings=warnings,
    )
    stride_status = _pick_stride_status(src_stride_status, dst_stride_status)

    requires_rebuild = layout_class in {"IRREGULAR", "BROADCAST_LIKE"}
    movement_status = "MOVEMENT_LEGAL"
    hardware_status = "HARDWARE_COMPATIBLE"

    if requires_rebuild:
        movement_status = "MOVEMENT_REBUILD_REQUIRED"
        hardware_status = "HARDWARE_UNSUPPORTED_PATTERN"
        suggestions.append(
            RepairSuggestion(
                type="REBUILD_LAYOUT",
                summary="Rebuild or materialize the tensor layout before moving data",
                rationale="The current stride pattern is not representable as a supported affine copy pattern.",
                expected_status_after_fix="VALID",
                patch_fields={"allow_repack": True},
            )
        )
        violations.append(
            RuleViolation(
                rule_id="LAYOUT_004",
                severity="error",
                message="layout is irregular or broadcast-like and requires rebuild before direct movement",
            )
        )
    elif layout_class == "TRANSPOSED_REGULAR":
        movement_status = "MOVEMENT_ROW_SPLIT_REQUIRED"
        hardware_status = "HARDWARE_UNSUPPORTED_PATTERN"
        suggestions.append(
            RepairSuggestion(
                type="SPLIT_ROW_COPY",
                summary="Rewrite the movement as row-wise or tiled row-wise copy",
                rationale="The layout is affine but not directly expressible as a whole-tensor contiguous move.",
                expected_status_after_fix="VALID_WITH_WARNING",
                patch_fields={"requested_copy_kind": "ROW_WISE"},
            )
        )
        violations.append(
            RuleViolation(
                rule_id="LAYOUT_003",
                severity="warning",
                message="layout is a regular permutation and needs row-wise staging instead of direct whole-tensor copy",
            )
        )
    elif stride_status in {"STRIDE_UNIT_MISMATCH", "STRIDE_VALUE_INVALID", "STRIDE_NOT_AFFINE"}:
        hardware_status = "HARDWARE_STRIDE_ENCODING_INVALID"
    elif derived_row_bytes % ub_alignment_bytes != 0:
        if requested_copy_kind == "PAD_COPY":
            movement_status = "MOVEMENT_LEGAL"
            hardware_status = "HARDWARE_ALIGNMENT_WARNING"
            warnings.append("copy payload is not naturally aligned; legality depends on pad-capable copy semantics")
        else:
            movement_status = "MOVEMENT_REQUIRES_PAD_COPY"
            hardware_status = "HARDWARE_ALIGNMENT_WARNING"
            suggestions.append(
                RepairSuggestion(
                    type="USE_PAD_COPY",
                    summary="Switch the move to a pad-capable copy",
                    rationale="The payload is not aligned to the required movement granularity, but the layout is regular.",
                    expected_status_after_fix="VALID_WITH_WARNING",
                    patch_fields={"requested_copy_kind": "PAD_COPY"},
                )
            )
            violations.append(
                RuleViolation(
                    rule_id="COPY_004",
                    severity="warning",
                    message="copy payload is not alignment-safe for plain copy and needs pad semantics",
                )
            )

    if use_row_view and derived_row_count <= 0:
        violations.append(
            RuleViolation(
                rule_id="COPY_001",
                severity="error",
                message="effective_row_count must be positive for row-wise movement",
            )
        )
        movement_status = "MOVEMENT_UNKNOWN"
        stride_status = _pick_stride_status(stride_status, "STRIDE_VALUE_INVALID")
    if derived_row_bytes <= 0:
        violations.append(
            RuleViolation(
                rule_id="COPY_002",
                severity="error",
                message="effective_row_bytes must be positive",
            )
        )
        movement_status = "MOVEMENT_UNKNOWN"
        stride_status = _pick_stride_status(stride_status, "STRIDE_VALUE_INVALID")

    if stride_status in {"STRIDE_VALUE_INVALID", "STRIDE_NOT_AFFINE"}:
        status = "INVALID"
        reason = "stride fields are malformed or incompatible with the inferred affine copy pattern"
    elif requires_rebuild:
        status = "REBUILD_REQUIRED"
        reason = "layout is not directly movable and requires rebuild before copy"
    elif movement_status in {"MOVEMENT_REQUIRES_PAD_COPY", "MOVEMENT_ROW_SPLIT_REQUIRED"} or stride_status in {"STRIDE_UNIT_MISMATCH"}:
        status = "REPAIRABLE"
        reason = "copy pattern is repairable with a small change to copy semantics or stride encoding"
    elif stride_status == "STRIDE_MISSING_INFERABLE" or hardware_status == "HARDWARE_ALIGNMENT_WARNING":
        status = "VALID_WITH_WARNING"
        reason = "request is legal after inferring missing copy metadata or honoring alignment-sensitive movement"
    else:
        status = "VALID"
        reason = "shape, layout, stride, and movement parameters are mutually consistent"

    normalized_request = {
        "logical_rank": len(shape),
        "total_elements": total_elements,
        "element_bytes": element_bytes,
        "element_dtype": dtype_name,
        "dense_stride": dense_stride,
        "logical_stride": stride,
        "movement_direction": direction,
        "src_space": src_space,
        "dst_space": dst_space,
        "requested_copy_kind": requested_copy_kind,
        "raw_src_stride": request.get("src_stride"),
        "raw_dst_stride": request.get("dst_stride"),
        "normalized_src_stride_bytes": src_stride_bytes,
        "normalized_dst_stride_bytes": dst_stride_bytes,
        "normalized_src_stride_blocks": src_stride_blocks,
        "normalized_dst_stride_blocks": dst_stride_blocks,
        "gm_alignment_bytes": gm_alignment_bytes,
        "ub_alignment_bytes": ub_alignment_bytes,
    }
    inferred_layout_details = {
        "tensor_shape": shape,
        "dense_stride": dense_stride,
        "logical_stride": stride,
        "inner_contiguous_span_elems": shape[-1] if shape and stride[-1] == 1 else 0,
        "inner_contiguous_span_bytes": (shape[-1] * element_bytes) if shape and stride[-1] == 1 else 0,
        "effective_row_count": derived_row_count,
        "effective_row_bytes": derived_row_bytes,
        "effective_src_gap_bytes": expected_src_gap_bytes,
        "effective_dst_gap_bytes": expected_dst_gap_bytes,
        "use_row_view": use_row_view,
    }

    return ShapeStrideLayoutValidatorResult(
        status=status,
        reason=reason,
        shape_status=shape_status,
        layout_class=layout_class,
        contiguity_status=contiguity_status,
        is_contiguous=is_contiguous,
        requires_rebuild=requires_rebuild,
        stride_status=stride_status,
        movement_status=movement_status,
        hardware_status=hardware_status,
        copy_param_status=_copy_param_status(status, movement_status, stride_status, requires_rebuild),
        normalized_request=normalized_request,
        inferred_layout_details=inferred_layout_details,
        violated_rules=violations,
        warnings=warnings,
        suggestions=suggestions,
        confidence=_confidence(stride_status, status, direction),
    )


__all__ = [
    "RuleViolation",
    "RepairSuggestion",
    "ShapeStrideLayoutValidatorResult",
    "plan_shape_stride_layout_validation",
]