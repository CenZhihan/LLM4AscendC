from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tools.common.error_extract import parse_layered_correctness_info

from .anchors import extract_anchor, normalize_anchor
from .manifest import build_manifest_lines
from .review_llm import _has_template_parts
from .schema import validate_record

# Extra CoT markers beyond review_llm template check.
_COT_MARKERS = (
    "following the pattern",
    "write exactly one",
    "we are given",
    "we need to",
)

# NL must contain at least one of these to count as "concrete".
_CONCRETE_NL_MARKERS = (
    "error:",
    "no member",
    "cannot convert",
    "valueerror",
    "parse_txt",
    "opbuild",
    "50703",
    "getoriginshape",
    "storageshape",
    "::",
    "`",
    "kernel_src",
    "host_operator",
    "pybind",
    "aclnn",
    "tiling",
)

_PACKAGING_ONLY_RE = re.compile(
    r"(cpack|cmake error|cmake_install|file install cannot find|install cannot find)",
    re.IGNORECASE,
)

_BUCKET_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("txt_missing_blocks", ("missing blocks", "parse_txt", "host_operator_src", "python_bind_src")),
    ("cpack_install", ("cpack", "cmake_install", "install cannot find", "binary/config")),
    ("opbuild", ("opbuild",)),
    ("aclnn_error_code", ("507035", "507034")),
    ("storage_shape_api", ("storageshape", "getoriginshape", "getdim", "shape*")),
    ("compile_api", ("error:", "no member named", "cannot convert", "kernel compilation error")),
    ("pybind_runtime", ("pybind", "pybind11", "dtype mismatch")),
)


@dataclass
class RemovedEntry:
    record: Dict[str, Any]
    phase: str  # "purge" | "dedup"
    reason: str
    bucket: str = ""

    def to_archive_line(self, *, removed_at: str) -> Dict[str, Any]:
        return {
            "removed_at": removed_at,
            "phase": self.phase,
            "reason": self.reason,
            "bucket": self.bucket,
            "memory_id": self.record.get("memory_id", ""),
            "record": self.record,
        }


@dataclass
class MaintenanceResult:
    kept: List[Dict[str, Any]] = field(default_factory=list)
    removed: List[RemovedEntry] = field(default_factory=list)
    buckets: Dict[str, List[str]] = field(default_factory=dict)
    dedup_rationales: Dict[str, str] = field(default_factory=dict)
    dedup_skipped_buckets: List[str] = field(default_factory=list)

    @property
    def purge_count(self) -> int:
        return sum(1 for r in self.removed if r.phase == "purge")

    @property
    def dedup_count(self) -> int:
        return sum(1 for r in self.removed if r.phase == "dedup")


def load_canonical_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _record_text_blob(record: Dict[str, Any]) -> str:
    parts = [
        record.get("natural_language") or "",
        record.get("root_cause_anchor_before") or "",
        record.get("root_cause_anchor_after") or "",
        record.get("symptom_anchor_before") or "",
        record.get("symptom_anchor_after") or "",
        record.get("error_anchors_before") or "",
        record.get("error_anchors_after") or "",
    ]
    return normalize_anchor(" ".join(parts))


def _root_anchor(record: Dict[str, Any]) -> str:
    root = (record.get("root_cause_anchor_after") or "").strip()
    if root:
        return root
    root_b = (record.get("root_cause_anchor_before") or "").strip()
    if root_b:
        return root_b
    ea = record.get("error_anchors_after") or ""
    eb = record.get("error_anchors_before") or ""
    parsed_root, _ = parse_layered_correctness_info(ea or eb)
    return parsed_root.strip()


def _symptom_anchor(record: Dict[str, Any]) -> str:
    sym = (record.get("symptom_anchor_after") or record.get("symptom_anchor_before") or "").strip()
    if sym:
        return sym
    ea = record.get("error_anchors_after") or ""
    eb = record.get("error_anchors_before") or ""
    _, parsed_sym = parse_layered_correctness_info(ea or eb)
    if parsed_sym.strip():
        return parsed_sym.strip()
    return extract_anchor(ea or eb)


def _has_cot_marker(nl: str) -> bool:
    tl = (nl or "").lower()
    return any(m in tl for m in _COT_MARKERS)


def _nl_has_concrete_detail(nl: str) -> bool:
    tl = (nl or "").lower()
    return any(m in tl for m in _CONCRETE_NL_MARKERS)


def _is_packaging_only_symptom(symptom: str) -> bool:
    s = (symptom or "").strip()
    if not s:
        return False
    if _PACKAGING_ONLY_RE.search(s) and not re.search(
        r"(error:\s*no member|cannot convert|kernel compilation|\.cpp:\d+:\d+:\s*error)",
        s,
        re.IGNORECASE,
    ):
        return True
    return False


def purge_reason(record: Dict[str, Any]) -> Optional[str]:
    """Return purge reason if record should be removed; else None."""
    if not validate_record(record):
        return "invalid_schema"
    nl = (record.get("natural_language") or "").strip()
    if not _has_template_parts(nl):
        return "invalid_nl_template"
    if _has_cot_marker(nl):
        return "cot_marker"
    root = _root_anchor(record)
    symptom = _symptom_anchor(record)
    if len(nl) >= 40 and not _nl_has_concrete_detail(nl):
        if not root.strip() and _is_packaging_only_symptom(symptom):
            return "low_detail"
    return None


def classify_bucket(record: Dict[str, Any]) -> str:
    blob = _record_text_blob(record)
    for bucket_id, keywords in _BUCKET_RULES:
        if any(kw in blob for kw in keywords):
            return bucket_id
    return "other"


def phase1_purge(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[RemovedEntry]]:
    kept: List[Dict[str, Any]] = []
    removed: List[RemovedEntry] = []
    for rec in records:
        reason = purge_reason(rec)
        if reason:
            removed.append(RemovedEntry(record=rec, phase="purge", reason=reason))
        else:
            kept.append(rec)
    return kept, removed


def _bucket_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[classify_bucket(rec)].append(rec)
    return dict(buckets)


def _parse_dedup_output(raw: str, *, valid_ids: Set[str]) -> Tuple[List[str], str, Optional[str]]:
    raw = (raw or "").strip()
    if not raw:
        return [], "", "empty_output"
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    i = 0
    n = len(raw)
    while i < n:
        j = raw.find("{", i)
        if j < 0:
            break
        try:
            obj, end = json.JSONDecoder().raw_decode(raw[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and "drop_ids" in obj:
            candidates.append((j, obj))
        i = j + max(end, 1)
    if not candidates:
        return [], "", "no_json_object"
    first_start, first_obj = candidates[0]
    obj = first_obj if not raw[:first_start].strip() else candidates[-1][1]
    drop_raw = obj.get("drop_ids")
    if not isinstance(drop_raw, list):
        return [], "", "drop_ids_not_list"
    drop_ids: List[str] = []
    for x in drop_raw:
        if isinstance(x, str) and x.strip() in valid_ids:
            drop_ids.append(x.strip())
    rationale = obj.get("rationale")
    r_text = rationale.strip() if isinstance(rationale, str) else ""
    return drop_ids, r_text, None


_DEDUP_SYSTEM = """You deduplicate repair memories within one error-class bucket.
Each manifest line is one memory (tab-separated: id, op, category, tool_mode, tier, root, symptom, summary).

Return a JSON object ONLY (no markdown fences):
{"drop_ids": ["..."], "rationale": "..."}

Rules:
- drop_ids: memory ids to REMOVE because they are redundant with another id in this bucket.
- Keep memories that teach a DISTINCT fix (different API, root cause, or approach) even if the bucket is large.
- Do NOT rewrite, merge, or summarize memories. Only list ids to drop.
- Every id in drop_ids MUST appear exactly as id=... on a manifest line below.
- drop_ids may be empty when nothing is redundant."""


def dedup_bucket_with_llm(
    *,
    llm_config: Dict[str, Any],
    bucket_id: str,
    records: List[Dict[str, Any]],
    llm_call: Optional[Callable[..., str]] = None,
) -> Tuple[List[str], str, Optional[str]]:
    """
    Ask LLM which ids to drop within a bucket.
    Returns (drop_ids, rationale, error_or_none).
    """
    if len(records) < 2:
        return [], "", None
    id_set = {str(r.get("memory_id", "")).strip() for r in records}
    id_set.discard("")
    lines = build_manifest_lines(records)
    manifest_text = "\n".join(lines)
    user = (
        f"Bucket: {bucket_id}\n"
        f"Remove redundant duplicate memories in this bucket only.\n\n"
        f"=== Manifest ===\n{manifest_text}\n"
    )
    if llm_call is not None:
        raw = llm_call(bucket_id=bucket_id, manifest_text=manifest_text, user=user)
    else:
        from .llm_util import assistant_message_text, openai_client_from_llm_config

        client = openai_client_from_llm_config(llm_config)
        model = llm_config.get("model") or ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DEDUP_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = assistant_message_text(resp.choices[0].message)
        except Exception as exc:  # noqa: BLE001
            return [], "", str(exc)
    drop_ids, rationale, err = _parse_dedup_output(raw, valid_ids=id_set)
    if err:
        return [], raw, err
    return drop_ids, rationale, None


def run_maintenance(
    records: List[Dict[str, Any]],
    *,
    llm_config: Optional[Dict[str, Any]] = None,
    skip_dedup: bool = False,
    llm_call: Optional[Callable[..., str]] = None,
) -> MaintenanceResult:
    result = MaintenanceResult()
    kept, purged = phase1_purge(records)
    result.removed.extend(purged)

    if skip_dedup or not kept:
        result.kept = kept
        result.buckets = {bid: [r.get("memory_id", "") for r in recs] for bid, recs in _bucket_records(kept).items()}
        return result

    buckets = _bucket_records(kept)
    result.buckets = {bid: [r.get("memory_id", "") for r in recs] for bid, recs in buckets.items()}
    drop_ids_all: Set[str] = set()

    for bucket_id, bucket_recs in buckets.items():
        if len(bucket_recs) < 2:
            continue
        if llm_config is None and llm_call is None:
            result.dedup_skipped_buckets.append(bucket_id)
            continue
        drop_ids, rationale, err = dedup_bucket_with_llm(
            llm_config=llm_config or {},
            bucket_id=bucket_id,
            records=bucket_recs,
            llm_call=llm_call,
        )
        if err:
            result.dedup_skipped_buckets.append(bucket_id)
            continue
        if rationale:
            result.dedup_rationales[bucket_id] = rationale
        for mid in drop_ids:
            if mid not in drop_ids_all:
                drop_ids_all.add(mid)
                rec = next((r for r in bucket_recs if r.get("memory_id") == mid), None)
                if rec is not None:
                    result.removed.append(
                        RemovedEntry(record=rec, phase="dedup", reason="llm_redundant", bucket=bucket_id)
                    )

    removed_ids = {e.record.get("memory_id") for e in result.removed}
    result.kept = [r for r in kept if r.get("memory_id") not in removed_ids]
    return result


def write_report(report_path: Path, result: MaintenanceResult, *, apply_mode: bool) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "apply_mode": apply_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kept_count": len(result.kept),
        "removed_count": len(result.removed),
        "purge_count": result.purge_count,
        "dedup_count": result.dedup_count,
        "kept_ids": [r.get("memory_id") for r in result.kept],
        "removed": [
            {
                "memory_id": e.record.get("memory_id"),
                "phase": e.phase,
                "reason": e.reason,
                "bucket": e.bucket,
                "op_key": e.record.get("op_key"),
                "natural_language": (e.record.get("natural_language") or "")[:300],
            }
            for e in result.removed
        ],
        "buckets": result.buckets,
        "dedup_rationales": result.dedup_rationales,
        "dedup_skipped_buckets": result.dedup_skipped_buckets,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_removed_archive(archive_path: Path, removed: List[RemovedEntry]) -> None:
    if not removed:
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    removed_at = datetime.now(timezone.utc).isoformat()
    with open(archive_path, "a", encoding="utf-8") as f:
        for entry in removed:
            f.write(json.dumps(entry.to_archive_line(removed_at=removed_at), ensure_ascii=False) + "\n")


def apply_canonical(canonical_path: Path, kept: List[Dict[str, Any]]) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    canonical_path.write_text(text, encoding="utf-8")


def load_report(report_path: Path) -> Dict[str, Any]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("report must be a JSON object")
    return data


def apply_from_report(
    records: List[Dict[str, Any]],
    report: Dict[str, Any],
    *,
    mode: str = "remove_list",
) -> Tuple[List[Dict[str, Any]], List[RemovedEntry], List[str]]:
    """
    Apply a frozen maintenance report without re-running purge/dedup.

    mode=remove_list (default): delete only memory_ids listed in report[\"removed\"].
      Safe when canonical gained new records after the report — they are kept.

    mode=kept_ids: keep only report[\"kept_ids\"] in original order.
      Use only when canonical has not changed since the report (no new append).
    """
    warnings: List[str] = []
    by_id = {str(r.get("memory_id", "")).strip(): r for r in records}
    by_id.pop("", None)

    if mode == "kept_ids":
        kept_ids = report.get("kept_ids") or []
        if not isinstance(kept_ids, list):
            raise ValueError("report kept_ids must be a list")
        want = [str(x).strip() for x in kept_ids if str(x).strip()]
        current_ids = set(by_id.keys())
        want_set = set(want)
        extra_in_canonical = sorted(current_ids - want_set)
        missing_in_canonical = sorted(want_set - current_ids)
        if extra_in_canonical:
            warnings.append(
                f"canonical has {len(extra_in_canonical)} id(s) not in report kept_ids "
                f"(will be removed): e.g. {extra_in_canonical[:3]}"
            )
        if missing_in_canonical:
            warnings.append(
                f"report lists {len(missing_in_canonical)} kept_id(s) absent from canonical "
                f"(skipped): e.g. {missing_in_canonical[:3]}"
            )
        kept = [by_id[mid] for mid in want if mid in by_id]
        removed_ids = current_ids - {r.get("memory_id") for r in kept}
    elif mode == "remove_list":
        removed_meta = report.get("removed") or []
        if not isinstance(removed_meta, list):
            raise ValueError("report removed must be a list")
        drop_ids: Set[str] = set()
        for item in removed_meta:
            if isinstance(item, dict):
                mid = str(item.get("memory_id", "")).strip()
                if mid:
                    drop_ids.add(mid)
        missing = sorted(drop_ids - set(by_id.keys()))
        if missing:
            warnings.append(
                f"{len(missing)} id(s) in report already absent from canonical: e.g. {missing[:3]}"
            )
        removed_ids = drop_ids & set(by_id.keys())
        kept = [r for r in records if r.get("memory_id") not in removed_ids]
    else:
        raise ValueError(f"unknown apply mode: {mode}")

    removed_entries: List[RemovedEntry] = []
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for item in report.get("removed") or []:
        if isinstance(item, dict) and item.get("memory_id"):
            meta_by_id[str(item["memory_id"])] = item

    for mid in removed_ids:
        rec = by_id.get(mid)
        if rec is None:
            continue
        meta = meta_by_id.get(mid, {})
        removed_entries.append(
            RemovedEntry(
                record=rec,
                phase=str(meta.get("phase") or "report"),
                reason=str(meta.get("reason") or "from_report"),
                bucket=str(meta.get("bucket") or ""),
            )
        )

    return kept, removed_entries, warnings
