from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "repair_memory_v1"


@dataclass
class RepairMemoryRecord:
    memory_id: str
    schema_version: str
    tier: str  # "A" or "B"
    confidence: str  # "high" or "medium"
    op_key: str
    category: str
    tool_mode: str
    strategy: str
    eval_mode: str
    transition: Dict[str, Any]
    failure_stage_before: str
    failure_stage_after: str
    error_anchors_before: str
    error_anchors_after: str
    code_digest_before: str
    code_digest_after: str
    natural_language: str
    symptom_anchor_before: str = ""
    symptom_anchor_after: str = ""
    root_cause_anchor_before: str = ""
    root_cause_anchor_after: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())


def validate_record(d: Dict[str, Any]) -> bool:
    if (d.get("schema_version") or "") != SCHEMA_VERSION:
        return False
    for k in (
        "memory_id",
        "tier",
        "op_key",
        "natural_language",
        "failure_stage_before",
        "failure_stage_after",
    ):
        if not (d.get(k) or "").strip():
            return False
    if str(d.get("tier") or "") not in ("A", "B"):
        return False
    if not isinstance(d.get("transition"), dict):
        return False
    return True
