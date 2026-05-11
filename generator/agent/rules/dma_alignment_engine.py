"""
DMA / alignment advisory for Ascend C data movement (CANN 8.3.rc baseline).

Aligns with kb102-style patterns: AscendC::DataCopy for UB-local moves;
prefer AscendC::DataCopyPad when GM↔UB extents or offsets are not 32B-aligned.

Does not import ApiDocRetriever — rules are static + computed checks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cann83_constants import (
    ALIGN_BYTES_CAST_DATACOPY_DEFAULT,
    ALIGN_BYTES_COMPARE_BLOCK,
    CANN_BASELINE_VERSION,
    dtype_elem_bytes,
)

ADVISORY_SCHEMA_VERSION = "1"

_DIRECTION_ALIASES = {
    "gm_to_ub": "gm_to_ub",
    "global_to_ub": "gm_to_ub",
    "gm2ub": "gm_to_ub",
    "ub_to_gm": "ub_to_gm",
    "ub2gm": "ub_to_gm",
    "ub_to_ub": "ub_to_ub",
    "l1_to_ub": "l1_to_ub",
    "ub_to_l1": "ub_to_l1",
    "gm_to_l1": "gm_to_l1",
    "l1_to_gm": "l1_to_gm",
}


def _norm_direction(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    k = str(raw).strip().lower().replace("-", "_")
    return _DIRECTION_ALIASES.get(k, k)


def _norm_dtype(raw: Optional[str]) -> str:
    return (raw or "float32").strip().lower()


def _transfer_bytes(elem_count: Optional[int], byte_length: Optional[int], dtype: str) -> int:
    if byte_length is not None:
        try:
            return max(int(byte_length), 0)
        except (TypeError, ValueError):
            return 0
    if elem_count is None:
        return 0
    try:
        ec = int(elem_count)
    except (TypeError, ValueError):
        return 0
    return ec * dtype_elem_bytes(dtype)


def _aligned(n: int, align: int) -> bool:
    return align > 0 and n % align == 0


def analyze_dma_alignment(query: str, args: Dict[str, Any]) -> Dict[str, Any]:
    chip = str(args.get("chip") or "DAV_2201")
    transfers_in = args.get("transfers")
    issues: List[Dict[str, str]] = []
    recommendations: List[str] = []
    per_transfer: List[Dict[str, Any]] = []
    parse_warnings: List[str] = []

    if not isinstance(transfers_in, list) or not transfers_in:
        parse_warnings.append("missing_or_empty_transfers: pass args.transfers=[{...}] for structured checks")

    transfers: List[Dict[str, Any]] = transfers_in if isinstance(transfers_in, list) else []

    for i, t in enumerate(transfers):
        if not isinstance(t, dict):
            parse_warnings.append(f"transfers[{i}] is not an object; skipped")
            continue
        direction = _norm_direction(t.get("direction"))
        dtype = _norm_dtype(t.get("dtype"))
        elem_count = t.get("elem_count")
        byte_length = t.get("byte_length")
        gm_off = t.get("gm_offset_bytes")
        prefer_raw = str(t.get("prefer_api") or "auto").strip()
        prefer_api = prefer_raw.lower().replace("ascendc::", "")
        involves = str(t.get("involves_api") or "").strip()

        total_b = _transfer_bytes(
            elem_count if elem_count is not None else None,
            byte_length if byte_length is not None else None,
            dtype,
        )

        try:
            off_b = int(gm_off) if gm_off is not None else 0
        except (TypeError, ValueError):
            off_b = 0

        row: Dict[str, Any] = {
            "index": i,
            "direction": direction,
            "dtype": dtype,
            "total_bytes": total_b,
            "gm_offset_bytes": off_b,
            "prefer_api": prefer_api,
        }

        api_pick = "auto"
        sev: Optional[str] = None
        code: Optional[str] = None
        detail: Optional[str] = None

        if total_b <= 0:
            issues.append(
                {
                    "severity": "warning",
                    "code": "unknown_extent",
                    "detail": f"transfer[{i}] missing elem_count/byte_length — cannot verify alignment.",
                }
            )
            row["suggested_api"] = None
            per_transfer.append(row)
            continue

        align32_ok = _aligned(total_b, ALIGN_BYTES_CAST_DATACOPY_DEFAULT) and _aligned(
            off_b, ALIGN_BYTES_CAST_DATACOPY_DEFAULT
        )

        if involves.lower() == "compare":
            cmp_ok = _aligned(total_b, ALIGN_BYTES_COMPARE_BLOCK)
            row["compare_256b_aligned"] = cmp_ok
            if not cmp_ok:
                sev = "error"
                code = "compare_not_256b"
                detail = (
                    f"Compare extent {total_b}B not multiple of {ALIGN_BYTES_COMPARE_BLOCK}B "
                    f"(count*sizeof(dtype) must align to 256B)."
                )
                recommendations.append(
                    f"Transfer[{i}]: pad count so total bytes % {ALIGN_BYTES_COMPARE_BLOCK} == 0 before Compare."
                )
        else:
            row["compare_256b_aligned"] = None

        if direction == "gm_to_ub":
            if align32_ok:
                api_pick = "DataCopy"
                row["note"] = "GM→UB extent and offset 32B-aligned: DataCopy may suffice (still validate edge tiles)."
            else:
                api_pick = "DataCopyPad"
                sev = sev or "warning"
                code = code or "prefer_datacopy_pad_gm_ub"
                detail = detail or (
                    "GM→UB move not 32B-aligned on extent/offset — kb102-style kernels often use DataCopyPad "
                    "for tail tiles."
                )
                recommendations.append(
                    f"Transfer[{i}]: use AscendC::DataCopyPad for non-aligned GM→UB slices; set padSize appropriately."
                )

        elif direction == "ub_to_gm":
            if not align32_ok:
                sev = sev or "warning"
                code = code or "ub_to_gm_align"
                detail = detail or (
                    "UB→GM write extent/GM offset not 32B-aligned — prefer padding tail tiles or adjust offset."
                )
                api_pick = "DataCopyPad"
                recommendations.append(
                    f"Transfer[{i}]: AscendC::DataCopyPad often used when GM slice start/size is not 32B-aligned."
                )
            else:
                api_pick = "DataCopy"
                row["note"] = "UB→GM with 32B-aligned extent/offset: DataCopy pattern common in kb102 bundles."
        elif direction in ("ub_to_ub", "l1_to_ub", "ub_to_l1"):
            api_pick = "DataCopy"
            row["note"] = "On-chip moves: match LocalTensor element layout; 32B alignment still recommended for vector APIs."
        else:
            row["note"] = "Specify direction (e.g. gm_to_ub) for stronger API suggestions."
            api_pick = "DataCopy" if align32_ok else "DataCopyPad"

        if prefer_api in ("datacopy", "datacopypad"):
            api_pick = "DataCopy" if prefer_api == "datacopy" else "DataCopyPad"

        row["suggested_api"] = api_pick

        if sev and code and detail:
            issues.append({"severity": sev, "code": code, "detail": detail})

        per_transfer.append(row)

    confidence = "high" if transfers and not parse_warnings else ("low" if not transfers else "medium")

    summary = f"chip={chip}, transfers={len(per_transfer)}, issues={len(issues)}"

    return {
        "schema_version": ADVISORY_SCHEMA_VERSION,
        "tool": "dma_alignment_engine",
        "cann_version": CANN_BASELINE_VERSION,
        "chip": chip,
        "summary": summary,
        "transfers": per_transfer,
        "issues": issues,
        "recommendations": recommendations
        or [
            "Verify 32B alignment for GM tensor slices used with DataCopy/DataCopyPad (CANN 8.3.rc baseline).",
            "For Compare, enforce 256B alignment on compared extents per API constraints.",
        ],
        "parse_warnings": parse_warnings,
        "alignment_confidence": confidence,
        "query_echo": (query or "")[:500],
    }

