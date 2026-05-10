from dataclasses import dataclass, field
from typing import Dict, List, Optional


NUMERIC_OK_STATUS = "numeric_ok"
PLANNER_OK_STATUS = "planner_ok"
VALIDATABLE_TILING_STATUSES = {"ok", NUMERIC_OK_STATUS}


@dataclass
class TilingParamsResult:
    """Result of tiling parameter computation."""

    status: str
    supported: bool
    operator_class: str
    strategy_kind: str
    reason: str
    required_inputs: List[str] = field(default_factory=list)
    block_num: Optional[int] = None
    num_per_core: Optional[int] = None
    tail_num_last_core: Optional[int] = None
    tile_length: Optional[int] = None
    repeat_times: Optional[int] = None
    ub_usage_bytes: Optional[int] = None
    ub_usage_pct: Optional[float] = None
    formula_used: Optional[str] = None
    constraints_met: Optional[bool] = None
    algorithm_kind: Optional[str] = None
    load_mode: Optional[str] = None
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
    warnings: List[str] = field(default_factory=list)


@dataclass
class TilingValidationResult:
    """Result of tiling parameter validation."""

    status: str
    is_valid: Optional[bool]
    reason: str
    errors: List[str]
    warnings: List[str]
    checks: Dict[str, bool]