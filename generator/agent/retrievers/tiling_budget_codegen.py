from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tiling_common import ceil_align, dtype_bytes, normalize_shape, num_elements
from .tiling_constants import ALIGNMENT_BYTES, DEFAULT_UB_CAPACITY, MAX_REPEAT_TIMES
from .tiling_validation import validate_tiling_params


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right if right > 0 else 0


def _align_down(value: int, align: int) -> int:
    if value <= 0 or align <= 0:
        return 0
    return (value // align) * align


def _chip_default_ub_bytes(chip: str) -> int:
    return 253952 if str(chip or "").strip().upper() == "DAV_3510" else DEFAULT_UB_CAPACITY


def _elements_per_repeat(elem_size: int) -> int:
    if elem_size <= 2:
        return 128
    if elem_size <= 4:
        return 64
    return 32


@dataclass
class NormalizedBudgetStage:
    stage_name: str
    position: str
    buffer_role: str
    allocation_kind: str
    per_tile_elements: int
    depth: int
    fixed_bytes: int
    dtype: str
    enable_double_buffer: bool
    member_name: str


@dataclass
class UBBudgetItem:
    stage_name: str
    position: str
    buffer_role: str
    allocation_kind: str
    bytes_per_buffer: int
    depth: int
    total_bytes: int
    alignment_bytes: int
    formula: str
    notes: str = ""


@dataclass
class TilingBudgetCodegenResult:
    status: str
    reason: str = ""
    required_inputs: List[str] = field(default_factory=list)
    supported: Optional[bool] = None
    operator_class: str = ""
    strategy_kind: str = ""
    algorithm_kind: Optional[str] = None
    load_mode: Optional[str] = None
    block_num: Optional[int] = None
    block_dim: Optional[int] = None
    tile_length: Optional[int] = None
    loop_count: Optional[int] = None
    tail_length: Optional[int] = None
    num_per_core: Optional[int] = None
    last_core_num: Optional[int] = None
    last_core_loop_count: Optional[int] = None
    last_core_tail_length: Optional[int] = None
    tail_num_last_core: Optional[int] = None
    repeat_times: Optional[int] = None
    stage_total_bytes: Optional[int] = None
    ub_reserved_bytes: Optional[int] = None
    ub_total_bytes: Optional[int] = None
    ub_usage_bytes: Optional[int] = None
    ub_usage_pct: Optional[float] = None
    output_count: Optional[int] = None
    output_elements: Optional[int] = None
    workspace_bytes: Optional[int] = None
    group_count: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_count: Optional[int] = None
    last_chunk_size: Optional[int] = None
    tile_a0_len: Optional[int] = None
    aligned_cols: Optional[int] = None
    collapsed_pattern: Optional[str] = None
    normalized_shape: List[int] = field(default_factory=list)
    normalized_axes: List[int] = field(default_factory=list)
    stage_summaries: List[str] = field(default_factory=list)
    formula_used: Optional[str] = None
    constraints_met: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)
    ub_budget_table: List[UBBudgetItem] = field(default_factory=list)
    init_code: str = ""
    strategy_suggestions: List[str] = field(default_factory=list)
    hardware_validation_status: str = "skipped"
    hardware_validation_errors: List[str] = field(default_factory=list)
    hardware_validation_warnings: List[str] = field(default_factory=list)
    planning_validation_status: str = "skipped"
    planning_validation_errors: List[str] = field(default_factory=list)
    planning_validation_warnings: List[str] = field(default_factory=list)
    seed_strategy_kind: str = ""
    seed_status: str = ""
    seed_result: Dict[str, Any] = field(default_factory=dict)


def _merge_unique_strings(*groups: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for item in group:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _seed_metadata(seed: Any) -> Dict[str, Any]:
    if seed is None:
        return {
            "supported": None,
            "strategy_kind": "",
            "algorithm_kind": None,
            "load_mode": None,
            "output_count": None,
            "output_elements": None,
            "workspace_bytes": None,
            "group_count": None,
            "chunk_size": None,
            "chunk_count": None,
            "last_chunk_size": None,
            "tile_a0_len": None,
            "aligned_cols": None,
            "collapsed_pattern": None,
            "normalized_shape": [],
            "normalized_axes": [],
            "stage_summaries": [],
            "formula_used": None,
            "seed_status": "",
            "seed_strategy_kind": "",
            "seed_result": {},
            "seed_warnings": [],
        }

    seed_result = {
        "status": str(getattr(seed, "status", "") or ""),
        "supported": getattr(seed, "supported", None),
        "operator_class": str(getattr(seed, "operator_class", "") or ""),
        "strategy_kind": str(getattr(seed, "strategy_kind", "") or ""),
        "reason": str(getattr(seed, "reason", "") or ""),
        "required_inputs": list(getattr(seed, "required_inputs", []) or []),
        "block_num": getattr(seed, "block_num", None),
        "num_per_core": getattr(seed, "num_per_core", None),
        "tail_num_last_core": getattr(seed, "tail_num_last_core", None),
        "tile_length": getattr(seed, "tile_length", None),
        "repeat_times": getattr(seed, "repeat_times", None),
        "ub_usage_bytes": getattr(seed, "ub_usage_bytes", None),
        "ub_usage_pct": getattr(seed, "ub_usage_pct", None),
        "formula_used": getattr(seed, "formula_used", None),
        "constraints_met": getattr(seed, "constraints_met", None),
        "algorithm_kind": getattr(seed, "algorithm_kind", None),
        "load_mode": getattr(seed, "load_mode", None),
        "output_count": getattr(seed, "output_count", None),
        "output_elements": getattr(seed, "output_elements", None),
        "workspace_bytes": getattr(seed, "workspace_bytes", None),
        "group_count": getattr(seed, "group_count", None),
        "chunk_size": getattr(seed, "chunk_size", None),
        "chunk_count": getattr(seed, "chunk_count", None),
        "last_chunk_size": getattr(seed, "last_chunk_size", None),
        "tile_a0_len": getattr(seed, "tile_a0_len", None),
        "aligned_cols": getattr(seed, "aligned_cols", None),
        "collapsed_pattern": getattr(seed, "collapsed_pattern", None),
        "normalized_shape": list(getattr(seed, "normalized_shape", []) or []),
        "normalized_axes": list(getattr(seed, "normalized_axes", []) or []),
        "stage_summaries": list(getattr(seed, "stage_summaries", []) or []),
        "warnings": list(getattr(seed, "warnings", []) or []),
    }
    return {
        "supported": seed_result["supported"],
        "strategy_kind": seed_result["strategy_kind"],
        "algorithm_kind": seed_result["algorithm_kind"],
        "load_mode": seed_result["load_mode"],
        "output_count": seed_result["output_count"],
        "output_elements": seed_result["output_elements"],
        "workspace_bytes": seed_result["workspace_bytes"],
        "group_count": seed_result["group_count"],
        "chunk_size": seed_result["chunk_size"],
        "chunk_count": seed_result["chunk_count"],
        "last_chunk_size": seed_result["last_chunk_size"],
        "tile_a0_len": seed_result["tile_a0_len"],
        "aligned_cols": seed_result["aligned_cols"],
        "collapsed_pattern": seed_result["collapsed_pattern"],
        "normalized_shape": seed_result["normalized_shape"],
        "normalized_axes": seed_result["normalized_axes"],
        "stage_summaries": seed_result["stage_summaries"],
        "formula_used": seed_result["formula_used"],
        "seed_status": seed_result["status"],
        "seed_strategy_kind": seed_result["strategy_kind"],
        "seed_result": seed_result,
        "seed_warnings": seed_result["warnings"],
    }


def _infer_total_elements(args: Dict[str, Any]) -> int:
    total_elements = _safe_int(args.get("total_elements"), 0)
    if total_elements > 0:
        return total_elements

    for key in ("total_shape", "output_shape", "input_shape"):
        shape = normalize_shape(args.get(key))
        if shape:
            return num_elements(shape)

    input_shapes = args.get("input_shapes")
    if isinstance(input_shapes, (list, tuple)) and len(input_shapes) == 1:
        shape = normalize_shape(input_shapes[0])
        if shape:
            return num_elements(shape)
    return 0


def _sanitize_member_name(stage_name: str) -> str:
    raw = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(stage_name or "stage"))
    raw = raw.strip("_") or "stage"
    if raw[0].isdigit():
        raw = f"stage_{raw}"
    return f"{raw}_"


def _default_position_for_role(buffer_role: str) -> str:
    if buffer_role == "input":
        return "VECIN"
    if buffer_role == "output":
        return "VECOUT"
    return "VECCALC"


def _default_depth_for_role(buffer_role: str, enable_double_buffer: bool) -> int:
    if buffer_role in {"input", "output"} and enable_double_buffer:
        return 2
    return 1


def _build_default_pipeline_stages(
    input_tensor_count: int,
    output_tensor_count: int,
    enable_double_buffer: bool,
    dtype: str,
) -> List[NormalizedBudgetStage]:
    stages: List[NormalizedBudgetStage] = []
    for index in range(max(0, input_tensor_count)):
        stage_name = f"in_{index}"
        stages.append(
            NormalizedBudgetStage(
                stage_name=stage_name,
                position="VECIN",
                buffer_role="input",
                allocation_kind="TQue",
                per_tile_elements=1,
                depth=_default_depth_for_role("input", enable_double_buffer),
                fixed_bytes=0,
                dtype=dtype,
                enable_double_buffer=enable_double_buffer,
                member_name=_sanitize_member_name(stage_name),
            )
        )
    for index in range(max(0, output_tensor_count)):
        stage_name = f"out_{index}"
        stages.append(
            NormalizedBudgetStage(
                stage_name=stage_name,
                position="VECOUT",
                buffer_role="output",
                allocation_kind="TQue",
                per_tile_elements=1,
                depth=_default_depth_for_role("output", enable_double_buffer),
                fixed_bytes=0,
                dtype=dtype,
                enable_double_buffer=enable_double_buffer,
                member_name=_sanitize_member_name(stage_name),
            )
        )
    return stages


def _normalize_pipeline_stages(
    args: Dict[str, Any],
    dtype: str,
    enable_double_buffer: bool,
) -> List[NormalizedBudgetStage]:
    raw_stages = args.get("pipeline_stages")
    if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
        return _build_default_pipeline_stages(
            input_tensor_count=_safe_int(args.get("input_tensor_count"), 0),
            output_tensor_count=_safe_int(args.get("output_tensor_count"), 0),
            enable_double_buffer=enable_double_buffer,
            dtype=dtype,
        )

    stages: List[NormalizedBudgetStage] = []
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            continue
        buffer_role = str(raw_stage.get("buffer_role") or "temp").strip().lower() or "temp"
        stage_db = _safe_bool(raw_stage.get("enable_double_buffer"), enable_double_buffer)
        default_depth = _default_depth_for_role(buffer_role, stage_db)
        stage_name = str(raw_stage.get("stage_name") or f"stage_{index}").strip() or f"stage_{index}"
        position = str(raw_stage.get("position") or _default_position_for_role(buffer_role)).strip().upper()
        allocation_kind = str(raw_stage.get("allocation_kind") or "TQue").strip() or "TQue"
        depth = max(1, _safe_int(raw_stage.get("depth"), default_depth))
        fixed_bytes = max(0, _safe_int(raw_stage.get("fixed_bytes"), 0))
        per_tile_default = 0 if fixed_bytes > 0 else 1
        per_tile_elements = max(0, _safe_int(raw_stage.get("per_tile_elements"), per_tile_default))
        stage_dtype = str(raw_stage.get("dtype") or dtype).strip() or dtype
        stages.append(
            NormalizedBudgetStage(
                stage_name=stage_name,
                position=position,
                buffer_role=buffer_role,
                allocation_kind=allocation_kind,
                per_tile_elements=per_tile_elements,
                depth=depth,
                fixed_bytes=fixed_bytes,
                dtype=stage_dtype,
                enable_double_buffer=stage_db,
                member_name=_sanitize_member_name(stage_name),
            )
        )
    return stages


def normalize_budget_codegen_request(args: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(args or {})
    chip = str(raw.get("chip") or "DAV_2201").strip() or "DAV_2201"
    dtype = str(raw.get("dtype") or "float16").strip() or "float16"
    enable_double_buffer = _safe_bool(raw.get("enable_double_buffer"), True)
    ub_total_bytes = max(1, _safe_int(raw.get("ub_total_bytes"), _chip_default_ub_bytes(chip)))
    ub_reserved_bytes = max(0, _safe_int(raw.get("ub_reserved_bytes"), 0))
    pipeline_stages = _normalize_pipeline_stages(raw, dtype, enable_double_buffer)
    total_elements = _infer_total_elements(raw)

    required_inputs: List[str] = []
    if total_elements <= 0:
        required_inputs.append("total_shape or total_elements")
    if not pipeline_stages:
        required_inputs.append("pipeline_stages or input/output tensor counts")

    return {
        "op_name": str(raw.get("op_name") or "").strip(),
        "op_type": str(raw.get("op_type") or "elementwise").strip() or "elementwise",
        "state_category": str(raw.get("state_category") or "").strip(),
        "query": str(raw.get("query") or "").strip(),
        "chip": chip,
        "dtype": dtype,
        "total_elements": total_elements,
        "total_shape": normalize_shape(raw.get("total_shape")),
        "input_shape": raw.get("input_shape"),
        "output_shape": raw.get("output_shape"),
        "input_shapes": raw.get("input_shapes"),
        "reduction_axes": raw.get("reduction_axes"),
        "keepdim": raw.get("keepdim"),
        "permutation": raw.get("permutation"),
        "track_index": raw.get("track_index"),
        "output_count": raw.get("output_count"),
        "algorithm_hint": raw.get("algorithm_hint"),
        "precision_sensitive": raw.get("precision_sensitive"),
        "ub_total_bytes": ub_total_bytes,
        "ub_reserved_bytes": ub_reserved_bytes,
        "enable_double_buffer": enable_double_buffer,
        "pipeline_stages": pipeline_stages,
        "required_inputs": required_inputs,
    }


def _estimate_budget_items(
    tile_length: int,
    stages: List[NormalizedBudgetStage],
    alignment_bytes: int,
) -> tuple[List[UBBudgetItem], int]:
    items: List[UBBudgetItem] = []
    stage_total_bytes = 0
    for stage in stages:
        elem_size = dtype_bytes(stage.dtype)
        dynamic_bytes = max(0, stage.per_tile_elements) * tile_length * elem_size
        raw_bytes = dynamic_bytes + max(0, stage.fixed_bytes)
        bytes_per_buffer = ceil_align(max(raw_bytes, 1), alignment_bytes)
        total_bytes = bytes_per_buffer * max(1, stage.depth)
        notes = "double buffer enabled" if stage.depth > 1 else "single buffer"
        if stage.allocation_kind != "TQue":
            notes = f"{notes}; non-TQue allocation requested"
        formula_parts: List[str] = []
        if stage.per_tile_elements > 0:
            formula_parts.append(
                f"align_up(tile_length * {stage.per_tile_elements} * sizeof({stage.dtype}), {alignment_bytes})"
            )
        if stage.fixed_bytes > 0:
            formula_parts.append(f"+ fixed_bytes({stage.fixed_bytes})")
        formula = " ".join(formula_parts) if formula_parts else f"align_up(1, {alignment_bytes})"
        items.append(
            UBBudgetItem(
                stage_name=stage.stage_name,
                position=stage.position,
                buffer_role=stage.buffer_role,
                allocation_kind=stage.allocation_kind,
                bytes_per_buffer=bytes_per_buffer,
                depth=stage.depth,
                total_bytes=total_bytes,
                alignment_bytes=alignment_bytes,
                formula=formula,
                notes=notes,
            )
        )
        stage_total_bytes += total_bytes
    return items, stage_total_bytes


def _build_init_code(stages: List[NormalizedBudgetStage], items: List[UBBudgetItem]) -> str:
    declarations = ["AscendC::TPipe pipe_;"]
    init_lines: List[str] = []
    for stage, item in zip(stages, items):
        if stage.allocation_kind == "TBuf":
            declarations.append(
                f"AscendC::TBuf<AscendC::TPosition::{stage.position}> {stage.member_name}"
                ";"
            )
            init_lines.append(f"pipe_.InitBuffer({stage.member_name}, {item.bytes_per_buffer});")
            continue
        declarations.append(
            f"AscendC::TQue<AscendC::TPosition::{stage.position}, {item.depth}> {stage.member_name};"
        )
        init_lines.append(
            f"pipe_.InitBuffer({stage.member_name}, {item.depth}, {item.bytes_per_buffer});"
        )
    return "\n".join(declarations + [""] + init_lines)


def _compute_loop_fields(
    main_core_num: int,
    last_core_num: int,
    tile_length: int,
) -> tuple[int, int, int, int]:
    loop_count = _ceil_div(main_core_num, tile_length)
    tail_length = (
        main_core_num - (loop_count - 1) * tile_length if loop_count > 0 else 0
    )
    last_core_loop_count = _ceil_div(last_core_num, tile_length)
    last_core_tail_length = (
        last_core_num - (last_core_loop_count - 1) * tile_length if last_core_loop_count > 0 else 0
    )
    return loop_count, tail_length, last_core_loop_count, last_core_tail_length


def _validate_planning(
    *,
    items: List[UBBudgetItem],
    stages: List[NormalizedBudgetStage],
    stage_total_bytes: int,
    ub_total_bytes: int,
    ub_reserved_bytes: int,
    ub_usage_bytes: int,
    init_code: str,
    loop_count: int,
    tail_length: int,
    elements_per_alignment: int,
) -> tuple[str, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    recomputed_total = sum(item.total_bytes for item in items)
    if recomputed_total != stage_total_bytes:
        errors.append(
            f"stage_total_bytes mismatch: expected {stage_total_bytes}, recomputed {recomputed_total}"
        )

    usable_stage_bytes = ub_total_bytes - ub_reserved_bytes
    if usable_stage_bytes <= 0:
        errors.append(
            f"ub_reserved_bytes = {ub_reserved_bytes} leaves no stage budget under ub_total_bytes = {ub_total_bytes}"
        )
    if stage_total_bytes > usable_stage_bytes:
        errors.append(
            f"stage_total_bytes = {stage_total_bytes} exceeds usable stage budget = {usable_stage_bytes}"
        )
    if ub_usage_bytes != stage_total_bytes + ub_reserved_bytes:
        errors.append(
            f"ub_usage_bytes mismatch: got {ub_usage_bytes}, expected {stage_total_bytes + ub_reserved_bytes}"
        )
    if ub_usage_bytes > ub_total_bytes:
        errors.append(
            f"ub_usage_bytes = {ub_usage_bytes} exceeds ub_total_bytes = {ub_total_bytes}"
        )

    for stage, item in zip(stages, items):
        if item.depth > 1 and stage.buffer_role not in {"input", "output"}:
            errors.append(
                f"stage {stage.stage_name} uses depth={item.depth}; only input/output stages may be double-buffered"
            )
        if item.depth > 1 and not stage.enable_double_buffer:
            errors.append(
                f"stage {stage.stage_name} uses depth={item.depth} even though enable_double_buffer=false"
            )
        if item.allocation_kind == "TBuf" and item.depth != 1:
            errors.append(f"stage {stage.stage_name} uses TBuf with depth={item.depth}; TBuf should be single-buffered")
        if stage.member_name not in init_code:
            errors.append(f"init_code is missing member {stage.member_name}")

    if loop_count <= 1 and any(item.depth > 1 for item in items):
        warnings.append("loop_count == 1; double buffering offers limited overlap and can likely be disabled")
    if tail_length > 0 and tail_length % max(1, elements_per_alignment) != 0:
        warnings.append("tail_length is not alignment-safe; generate a dedicated tail path or use DataCopyPad")

    return ("ok" if not errors else "error"), errors, warnings


def _build_strategy_suggestions(
    *,
    items: List[UBBudgetItem],
    loop_count: int,
    tail_length: int,
    elements_per_alignment: int,
    ub_usage_pct: float,
) -> List[str]:
    suggestions: List[str] = []
    if tail_length > 0 and tail_length % max(1, elements_per_alignment) != 0:
        suggestions.append(
            "tail_length is not alignment-safe for the selected dtype; generate a dedicated tail path or use DataCopyPad."
        )
    else:
        suggestions.append(
            "tail_length is aligned and large enough; reuse the main loop body with a final short iteration."
        )

    has_double_buffer = any(item.depth > 1 for item in items)
    if has_double_buffer and loop_count >= 2 and ub_usage_pct <= 90.0:
        suggestions.append(
            "double buffer is beneficial because loop_count >= 2 and UB headroom is sufficient."
        )
    elif has_double_buffer:
        suggestions.append(
            "double buffer offers limited overlap for the current loop count; consider using depth=1 on input/output queues."
        )

    if ub_usage_pct < 30.0:
        suggestions.append(
            "UB usage is low; consider increasing per-tile staging or reducing reserved bytes if bandwidth remains the bottleneck."
        )
    elif ub_usage_pct > 90.0:
        suggestions.append(
            "UB usage is close to capacity; keep temp buffers single-buffered and avoid extra workspace stages."
        )
    return suggestions


def _failure_result(
    *,
    status: str,
    reason: str,
    required_inputs: List[str],
    operator_class: str,
    ub_total_bytes: int,
    ub_reserved_bytes: int,
    seed: Any = None,
) -> TilingBudgetCodegenResult:
    seed_meta = _seed_metadata(seed)
    return TilingBudgetCodegenResult(
        status=status,
        reason=reason,
        required_inputs=required_inputs,
        supported=seed_meta["supported"],
        operator_class=operator_class,
        strategy_kind=seed_meta["strategy_kind"],
        algorithm_kind=seed_meta["algorithm_kind"],
        load_mode=seed_meta["load_mode"],
        output_count=seed_meta["output_count"],
        output_elements=seed_meta["output_elements"],
        workspace_bytes=seed_meta["workspace_bytes"],
        group_count=seed_meta["group_count"],
        chunk_size=seed_meta["chunk_size"],
        chunk_count=seed_meta["chunk_count"],
        last_chunk_size=seed_meta["last_chunk_size"],
        tile_a0_len=seed_meta["tile_a0_len"],
        aligned_cols=seed_meta["aligned_cols"],
        collapsed_pattern=seed_meta["collapsed_pattern"],
        normalized_shape=seed_meta["normalized_shape"],
        normalized_axes=seed_meta["normalized_axes"],
        stage_summaries=seed_meta["stage_summaries"],
        formula_used=seed_meta["formula_used"],
        ub_total_bytes=ub_total_bytes,
        ub_reserved_bytes=ub_reserved_bytes,
        constraints_met=False,
        warnings=seed_meta["seed_warnings"],
        seed_status=seed_meta["seed_status"],
        seed_strategy_kind=seed_meta["seed_strategy_kind"],
        seed_result=seed_meta["seed_result"],
        planning_validation_status="skipped",
        hardware_validation_status="skipped",
    )


def plan_tiling_budget_codegen(
    request: Dict[str, Any],
    *,
    tiling_retriever: Any = None,
) -> TilingBudgetCodegenResult:
    from .tiling_retriever import TilingRetriever

    normalized = normalize_budget_codegen_request(request)
    if normalized["required_inputs"]:
        return _failure_result(
            status="invalid_request",
            reason="tiling budget/codegen requires structured workload and pipeline information",
            required_inputs=normalized["required_inputs"],
            operator_class="",
            ub_total_bytes=normalized["ub_total_bytes"],
            ub_reserved_bytes=normalized["ub_reserved_bytes"],
        )

    if tiling_retriever is None:
        tiling_retriever = TilingRetriever()

    chip = normalized["chip"]
    dtype = normalized["dtype"]
    op_type = normalized["op_type"]
    total_elements = normalized["total_elements"]
    ub_total_bytes = normalized["ub_total_bytes"]
    ub_reserved_bytes = normalized["ub_reserved_bytes"]
    pipeline_stages = normalized["pipeline_stages"]
    usable_stage_ub_bytes = max(ALIGNMENT_BYTES, ub_total_bytes - ub_reserved_bytes)

    seed = tiling_retriever.compute_tiling(
        total_elements=total_elements,
        dtype=dtype,
        op_type=op_type,
        ub_capacity_bytes=usable_stage_ub_bytes,
        op_name=normalized["op_name"],
        state_category=normalized["state_category"],
        query=normalized["query"],
        input_shapes=normalized["input_shapes"],
        input_shape=normalized["input_shape"],
        output_shape=normalized["output_shape"],
        permutation=normalized["permutation"],
        reduction_axes=normalized["reduction_axes"],
        keepdim=normalized["keepdim"],
        track_index=normalized["track_index"],
        output_count=normalized["output_count"],
        algorithm_hint=normalized["algorithm_hint"],
        precision_sensitive=normalized["precision_sensitive"],
        chip=chip,
    )

    if seed.status == "unsupported_without_operator_specific_strategy":
        return _failure_result(
            status=seed.status,
            reason=seed.reason,
            required_inputs=list(seed.required_inputs or []),
            operator_class=seed.operator_class,
            ub_total_bytes=ub_total_bytes,
            ub_reserved_bytes=ub_reserved_bytes,
            seed=seed,
        )

    seed_meta = _seed_metadata(seed)

    operator_class = str(seed.operator_class or op_type or "elementwise")
    elem_size = dtype_bytes(dtype)
    alignment_bytes = 32 if operator_class == "reduction" else ALIGNMENT_BYTES
    elements_per_alignment = max(1, alignment_bytes // max(elem_size, 1))
    elements_per_repeat = _elements_per_repeat(elem_size)
    max_blocks = 32 if chip in ("DAV_2201", "DAV_1001", "DAV_2002", "DAV_3002") else 64
    base_block_dim = min(max_blocks, max(1, int(seed.block_num or min(max_blocks, total_elements))))

    block_candidates: List[int] = []
    for delta in range(0, 5):
        for sign in (0, -1, 1):
            candidate = base_block_dim if sign == 0 else base_block_dim + sign * delta
            if candidate < 1 or candidate > max_blocks or candidate > total_elements or candidate in block_candidates:
                continue
            block_candidates.append(candidate)

    best_candidate: Optional[Dict[str, Any]] = None
    best_score: Optional[tuple[int, int, int, int]] = None

    for block_dim in block_candidates:
        main_core_num = total_elements // block_dim
        if main_core_num <= 0:
            continue
        last_core_num = total_elements - main_core_num * (block_dim - 1)

        max_tile_by_repeat = max(elements_per_alignment, MAX_REPEAT_TIMES * elements_per_repeat)
        start_tile_hint = max(int(seed.tile_length or 0), min(main_core_num, max_tile_by_repeat))
        tile_length = _align_down(start_tile_hint, elements_per_alignment)
        if tile_length <= 0:
            tile_length = elements_per_alignment

        while tile_length >= elements_per_alignment:
            repeat_times = _ceil_div(tile_length, elements_per_repeat)
            if repeat_times > MAX_REPEAT_TIMES:
                tile_length -= elements_per_alignment
                continue

            budget_items, stage_total_bytes = _estimate_budget_items(
                tile_length=tile_length,
                stages=pipeline_stages,
                alignment_bytes=alignment_bytes,
            )
            ub_usage_bytes = stage_total_bytes + ub_reserved_bytes
            ub_usage_pct = (ub_usage_bytes / ub_total_bytes) * 100.0 if ub_total_bytes > 0 else 0.0
            loop_count, tail_length, last_core_loop_count, last_core_tail_length = _compute_loop_fields(
                main_core_num=main_core_num,
                last_core_num=last_core_num,
                tile_length=tile_length,
            )
            init_code = _build_init_code(pipeline_stages, budget_items)

            hardware_validation = validate_tiling_params(
                {
                    "operator_class": operator_class,
                    "tile_length": tile_length,
                    "repeat_times": repeat_times,
                    "ub_usage_bytes": ub_usage_bytes,
                    "block_num": block_dim,
                    "dtype": dtype,
                    "chip": chip,
                },
                chip=chip,
            )
            planning_status, planning_errors, planning_warnings = _validate_planning(
                items=budget_items,
                stages=pipeline_stages,
                stage_total_bytes=stage_total_bytes,
                ub_total_bytes=ub_total_bytes,
                ub_reserved_bytes=ub_reserved_bytes,
                ub_usage_bytes=ub_usage_bytes,
                init_code=init_code,
                loop_count=loop_count,
                tail_length=tail_length,
                elements_per_alignment=elements_per_alignment,
            )

            if hardware_validation.is_valid and planning_status == "ok":
                tiny_workload_penalty = max(0, elements_per_alignment - min(main_core_num, last_core_num))
                imbalance = abs(main_core_num - last_core_num)
                score = (min(block_dim, total_elements), -tiny_workload_penalty, -imbalance, tile_length)
                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = {
                        "block_dim": block_dim,
                        "tile_length": tile_length,
                        "loop_count": loop_count,
                        "tail_length": tail_length,
                        "num_per_core": main_core_num,
                        "last_core_num": last_core_num,
                        "last_core_loop_count": last_core_loop_count,
                        "last_core_tail_length": last_core_tail_length,
                        "tail_num_last_core": last_core_num,
                        "repeat_times": repeat_times,
                        "stage_total_bytes": stage_total_bytes,
                        "ub_usage_bytes": ub_usage_bytes,
                        "ub_usage_pct": round(ub_usage_pct, 1),
                        "ub_budget_table": budget_items,
                        "init_code": init_code,
                        "hardware_validation": hardware_validation,
                        "planning_status": planning_status,
                        "planning_errors": planning_errors,
                        "planning_warnings": planning_warnings,
                    }
                break

            tile_length -= elements_per_alignment

    if best_candidate is None:
        return _failure_result(
            status="no_valid_plan",
            reason="failed to find an alignment-safe tile that satisfies hardware validation and UB planning budget",
            required_inputs=[],
            operator_class=operator_class,
            ub_total_bytes=ub_total_bytes,
            ub_reserved_bytes=ub_reserved_bytes,
            seed=seed,
        )

    suggestions = _build_strategy_suggestions(
        items=best_candidate["ub_budget_table"],
        loop_count=best_candidate["loop_count"],
        tail_length=best_candidate["tail_length"],
        elements_per_alignment=elements_per_alignment,
        ub_usage_pct=float(best_candidate["ub_usage_pct"]),
    )
    suggestions.extend(best_candidate["planning_warnings"])

    hardware_validation = best_candidate["hardware_validation"]
    final_warnings = _merge_unique_strings(
        seed_meta["seed_warnings"],
        hardware_validation.warnings,
        best_candidate["planning_warnings"],
    )
    return TilingBudgetCodegenResult(
        status="ok",
        supported=seed_meta["supported"],
        operator_class=operator_class,
        strategy_kind=seed_meta["strategy_kind"],
        algorithm_kind=seed_meta["algorithm_kind"],
        load_mode=seed_meta["load_mode"],
        block_num=best_candidate["block_dim"],
        block_dim=best_candidate["block_dim"],
        tile_length=best_candidate["tile_length"],
        loop_count=best_candidate["loop_count"],
        tail_length=best_candidate["tail_length"],
        num_per_core=best_candidate["num_per_core"],
        last_core_num=best_candidate["last_core_num"],
        last_core_loop_count=best_candidate["last_core_loop_count"],
        last_core_tail_length=best_candidate["last_core_tail_length"],
        tail_num_last_core=best_candidate["tail_num_last_core"],
        repeat_times=best_candidate["repeat_times"],
        stage_total_bytes=best_candidate["stage_total_bytes"],
        ub_reserved_bytes=ub_reserved_bytes,
        ub_total_bytes=ub_total_bytes,
        ub_usage_bytes=best_candidate["ub_usage_bytes"],
        ub_usage_pct=best_candidate["ub_usage_pct"],
        output_count=seed_meta["output_count"],
        output_elements=seed_meta["output_elements"],
        workspace_bytes=seed_meta["workspace_bytes"],
        group_count=seed_meta["group_count"],
        chunk_size=seed_meta["chunk_size"],
        chunk_count=seed_meta["chunk_count"],
        last_chunk_size=seed_meta["last_chunk_size"],
        tile_a0_len=seed_meta["tile_a0_len"],
        aligned_cols=seed_meta["aligned_cols"],
        collapsed_pattern=seed_meta["collapsed_pattern"],
        normalized_shape=seed_meta["normalized_shape"],
        normalized_axes=seed_meta["normalized_axes"],
        stage_summaries=seed_meta["stage_summaries"],
        formula_used=seed_meta["formula_used"],
        constraints_met=hardware_validation.is_valid and best_candidate["planning_status"] == "ok",
        warnings=final_warnings,
        ub_budget_table=best_candidate["ub_budget_table"],
        init_code=best_candidate["init_code"],
        strategy_suggestions=suggestions,
        hardware_validation_status=hardware_validation.status,
        hardware_validation_errors=list(hardware_validation.errors or []),
        hardware_validation_warnings=list(hardware_validation.warnings or []),
        planning_validation_status=best_candidate["planning_status"],
        planning_validation_errors=best_candidate["planning_errors"],
        planning_validation_warnings=best_candidate["planning_warnings"],
        seed_strategy_kind=seed_meta["seed_strategy_kind"],
        seed_status=seed_meta["seed_status"],
        seed_result=seed_meta["seed_result"],
    )


__all__ = [
    "UBBudgetItem",
    "TilingBudgetCodegenResult",
    "normalize_budget_codegen_request",
    "plan_tiling_budget_codegen",
]