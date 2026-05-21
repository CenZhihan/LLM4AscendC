"""
Heuristic dtype / accumulation policy advice aligned with PyTorch reference patterns (CANN 8.3.rc).
Does not import retrievers — advisory only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .cann83_constants import CANN_BASELINE_VERSION

ADVISORY_SCHEMA_VERSION = "1"

_VALID_MODES = frozenset(
    {
        "match_pytorch",
        "fp32_accum_fp16_io",
        "fp32_accum_bf16_io",
        "strict_signature",
        "integer_exact",
    }
)

_VALID_FAMILIES = frozenset(
    {
        "elementwise",
        "matmul_like",
        "reduce",
        "conv_like",
        "norm_like",
        "mixed",
        "unknown",
    }
)


def _normalize_mode(raw: Optional[str]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    m = (raw or "").strip().lower() or "match_pytorch"
    if m not in _VALID_MODES:
        warnings.append(f"unknown target_precision_mode {raw!r}; falling back to match_pytorch")
        m = "match_pytorch"
    return m, warnings


def _normalize_family(raw: Optional[str]) -> str:
    f = (raw or "unknown").strip().lower()
    return f if f in _VALID_FAMILIES else "unknown"


def _infer_accum_dtype(mode: str, family: str, io_primary: str) -> Tuple[str, str]:
    """Returns (accum_dtype, rationale)."""
    if mode == "integer_exact":
        return io_primary, "integer_exact: keep accumulation in integer dtype unless API forces otherwise"

    if mode == "strict_signature":
        return io_primary, "strict_signature: avoid widening beyond I/O dtypes unless an API requires it"

    if mode == "fp32_accum_fp16_io":
        return "float32", "fp32_accum_fp16_io: FP32 accumulation with FP16-class I/O (PyTorch AMP-like)"

    if mode == "fp32_accum_bf16_io":
        return "float32", "fp32_accum_bf16_io: FP32 accumulation with BF16 I/O"

    # match_pytorch-style defaults
    if family in ("matmul_like", "conv_like", "norm_like"):
        return "float32", "match_pytorch: matmul/conv/norm stacks commonly accumulate in FP32"

    if family == "reduce":
        if io_primary in ("half", "float16", "bf16", "bfloat16"):
            return "float32", "match_pytorch: sum/mean-style reduce often upcasts to FP32"
        return io_primary, "match_pytorch: use same dtype as I/O for min/max class reduces when I/O is FP32"

    if family == "elementwise":
        return io_primary, "match_pytorch: elementwise unary/binary typically preserves compute dtype"

    return io_primary, "match_pytorch / heuristic: insufficient family hint; align casts with reference"


def _normalize_dtype_label(s: str) -> str:
    x = (s or "").strip().lower()
    aliases = {
        "fp16": "half",
        "float16": "half",
        "fp32": "float32",
        "float": "float32",
        "bf16": "bfloat16",
    }
    return aliases.get(x, x)


def _primary_io_from_args(io_dtypes: Any) -> str:
    if not isinstance(io_dtypes, dict):
        return "float32"
    io = io_dtypes.get("io")
    if isinstance(io, str) and io.strip():
        return _normalize_dtype_label(io)
    outs = io_dtypes.get("output")
    if isinstance(outs, str) and outs.strip():
        return _normalize_dtype_label(outs)
    inputs = io_dtypes.get("inputs")
    if isinstance(inputs, list) and inputs:
        first = inputs[0]
        if isinstance(first, str):
            return _normalize_dtype_label(first)
    return "float32"


def analyze_dtype_policy(query: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce structured dtype policy advice.

    Args:
        query: Free-text intent from tool JSON ``query`` field.
        args: Structured ``args`` from tool JSON (optional keys per plan).
    """
    mode, parse_warnings = _normalize_mode(args.get("target_precision_mode"))
    family = _normalize_family(args.get("op_family"))
    io_dtypes = args.get("io_dtypes")
    io_primary = _primary_io_from_args(io_dtypes)

    if mode == "integer_exact" and io_primary.startswith(("float", "half", "bf16", "fp")):
        parse_warnings.append("integer_exact conflicts with floating io_dtypes; recommendations assume integer reinterpretation was intended")

    accum_dtype, accum_rationale = _infer_accum_dtype(mode, family, io_primary)

    stages_out: List[Dict[str, Any]] = []
    for name in ("load", "compute", "accumulate", "store"):
        rec: Dict[str, Any] = {
            "stage": name,
            "tensor_roles": ["input", "output", "workspace"],
            "recommended_compute_dtype": accum_dtype,
        }
        if name == "load":
            rec["recommended_compute_dtype"] = io_primary
            rec["note"] = (
                "Match GM/UB tensor dtypes to Cast/DataCopy element types; Cast up to compute dtype before math."
            )
        elif name == "compute":
            rec["recommended_compute_dtype"] = accum_dtype
            rec["note"] = "Vector pipe dtype should match accumulation policy for the active tile."
        elif name == "accumulate":
            rec["recommended_compute_dtype"] = accum_dtype
            rec["note"] = accum_rationale
        elif name == "store":
            rec["recommended_compute_dtype"] = io_primary
            rec["note"] = "Cast back to reference output dtype before UB→GM or final write."
        stages_out.append(rec)

    issues: List[Dict[str, str]] = []
    recommendations: List[str] = [
        f"Prefer accumulation dtype {accum_dtype} for family={family} under mode={mode}.",
        "Cross-check with vendor/mkb/reference/<category>/<op>.py dtypes when generating kernels.",
    ]

    if family == "unknown":
        issues.append(
            {
                "severity": "info",
                "code": "vague_op_family",
                "detail": "op_family is unknown; accumulation heuristic uses conservative defaults.",
            }
        )

    confidence = "high" if isinstance(io_dtypes, dict) and io_dtypes else "heuristic"

    summary = (
        f"mode={mode}, family={family}, io≈{io_primary}, accum→{accum_dtype} ({confidence})"
    )

    return {
        "schema_version": ADVISORY_SCHEMA_VERSION,
        "tool": "dtype_policy_engine",
        "cann_version": CANN_BASELINE_VERSION,
        "summary": summary,
        "target_precision_mode": mode,
        "op_family": family,
        "io_dtypes_hint": io_dtypes if isinstance(io_dtypes, dict) else None,
        "parse_warnings": parse_warnings,
        "stages": stages_out,
        "issues": issues,
        "recommendations": recommendations,
        "alignment_confidence": confidence,
        "query_echo": (query or "")[:500],
    }

