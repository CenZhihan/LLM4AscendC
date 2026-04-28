"""
API Documentation Retriever for Ascend C kernel development agent.

Provides API signature lookup, constraint checking, and alternative finding
based on Knowledge/api/ documentation files.

Three capabilities share a single retriever because they all parse the
same API documentation source:
1. lookup_signature()  — API signature, dtypes, repeatTimes limit
2. check_constraints() — Alignment, blockCount, platform constraints
3. find_alternatives() — Alternative APIs when primary is unavailable
"""
import os
import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Structured result types
# ============================================================

@dataclass
class ApiSignatureResult:
    """Result of API signature lookup."""
    api_name: str
    signature: str                          # Full function signature
    supported_dtypes: List[str]             # ["half", "float", ...]
    repeat_times_limit: Optional[int]       # 255 or None
    params: List[dict]                      # [{name, type, direction}, ...]
    example_call: str                       # Example usage
    source_doc: str                         # Source document path
    details: str                            # Human-readable summary
    header_files: List[str] = field(default_factory=list)
    doc_metadata: List[dict] = field(default_factory=list)
    match_kind: str = "not_found"
    confidence: str = "low"
    is_actionable: bool = False


@dataclass
class ApiConstraintResult:
    """Result of API constraint check."""
    api_name: str
    constraints: List[dict]                 # [{type, description, severity}]
    violations: List[str]                   # Current violations
    suggestion: str                         # Fix suggestion
    is_compliant: bool                      # Whether current call is compliant
    checked_context: Dict[str, object] = field(default_factory=dict)
    checks_performed: List[dict] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    source_doc: str = ""
    compliance_status: str = "pass"


@dataclass
class ApiAlternativeResult:
    """Result of API alternative lookup."""
    primary_api: str
    alternatives: List[dict]                # [{api, steps, performance_impact, precision_impact}]
    recommended: str                        # Recommended approach
    reason: str                             # Why alternatives are needed


@dataclass
class _HeaderSearchResult:
    header_files: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)
    signature: str = ""
    summary: str = ""


_COMMENT_PREFIXES = ("//", "/*", "*", "#")


def _is_non_signature_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return True
    if stripped.startswith(_COMMENT_PREFIXES):
        return True
    if stripped.startswith("|"):
        return True
    return False


# ============================================================
# Built-in API knowledge base
#
# Extracted from Knowledge/api/ documentation files.
# Contains common Ascend C Vector/Cube/Scalar API info.
# ============================================================

# Common dtype sets used across APIs
_DTYPES_ALL = ["float", "float16", "half", "int8_t", "int16_t", "int32_t",
               "uint8_t", "uint16_t", "uint32_t", "bfloat16_t"]
_DTYPES_FLOAT = ["float", "half", "bfloat16_t"]
_DTYPES_FLOAT_HIGH = ["float", "half"]
_DTYPES_INT = ["int8_t", "int16_t", "int32_t", "uint8_t", "uint16_t", "uint32_t"]

_API_KNOWLEDGE: Dict[str, dict] = {
    # ===== Arithmetic =====
    "Add": {
        "signature": "void Add(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src0", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "src1", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "mask", "type": "uint64_t", "dir": "in"},
            {"name": "repeatTime", "type": "uint8_t", "dir": "in"},
            {"name": "repeatParams", "type": "BinaryRepeatParams&", "dir": "in"},
        ],
        "example": "Add(dst, src0, src1, mask, repeatTimes, params)",
        "source": "api-arithmetic.md",
        "constraints": [
            {"type": "alignment", "desc": "数据需 32 字节对齐", "severity": "warning"},
        ],
    },
    "Sub": {
        "signature": "void Sub(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src0", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "src1", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "mask", "type": "uint64_t", "dir": "in"},
            {"name": "repeatTime", "type": "uint8_t", "dir": "in"},
            {"name": "repeatParams", "type": "BinaryRepeatParams&", "dir": "in"},
        ],
        "example": "Sub(dst, src0, src1, mask, repeatTimes, params)",
        "source": "api-arithmetic.md",
        "constraints": [
            {"type": "alignment", "desc": "数据需 32 字节对齐", "severity": "warning"},
        ],
    },
    "Mul": {
        "signature": "void Mul(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src0", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "src1", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "mask", "type": "uint64_t", "dir": "in"},
            {"name": "repeatTime", "type": "uint8_t", "dir": "in"},
            {"name": "repeatParams", "type": "BinaryRepeatParams&", "dir": "in"},
        ],
        "example": "Mul(dst, src0, src1, mask, repeatTimes, params)",
        "source": "api-arithmetic.md",
        "constraints": [],
    },
    "Div": {
        "signature": "void Div(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src0", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "src1", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "mask", "type": "uint64_t", "dir": "in"},
            {"name": "repeatTime", "type": "uint8_t", "dir": "in"},
            {"name": "repeatParams", "type": "BinaryRepeatParams&", "dir": "in"},
        ],
        "example": "Div(dst, src0, src1, mask, repeatTimes, params)",
        "source": "api-arithmetic.md",
        "constraints": [
            {"type": "data_size", "desc": "除数不能为 0", "severity": "error"},
        ],
    },
    "Adds": {
        "signature": "void Adds(LocalTensor<T>& dst, LocalTensor<T>& src, const T& scalarValue, int32_t count)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "scalarValue", "type": "const T&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Adds(dst, src, -maxVal, count)  // 减法转加法",
        "source": "api-arithmetic.md",
        "constraints": [],
    },
    "Muls": {
        "signature": "void Muls(LocalTensor<T>& dst, LocalTensor<T>& src, const T& scalarValue, int32_t count)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "scalarValue", "type": "const T&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Muls(dst, src, invSum, count)  // 除法转乘法",
        "source": "api-arithmetic.md",
        "constraints": [],
    },

    # ===== Unary Math =====
    "Exp": {
        "signature": "void Exp(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)",
        "dtypes": _DTYPES_FLOAT_HIGH,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Exp(dst, src, count)",
        "source": "api-restrictions.md",
        "constraints": [],
    },
    "Log": {
        "signature": "void Log(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)",
        "dtypes": _DTYPES_FLOAT_HIGH,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Log(dst, src, count)",
        "source": "api-restrictions.md",
        "constraints": [
            {"type": "data_size", "desc": "输入值需 > 0", "severity": "warning"},
        ],
    },
    "Sqrt": {
        "signature": "void Sqrt(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)",
        "dtypes": _DTYPES_FLOAT_HIGH,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Sqrt(dst, src, count)",
        "source": "api-restrictions.md",
        "constraints": [
            {"type": "data_size", "desc": "输入值需 >= 0", "severity": "warning"},
        ],
    },
    "Abs": {
        "signature": "void Abs(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)",
        "dtypes": _DTYPES_FLOAT + _DTYPES_INT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Abs(dst, src, count)",
        "source": "api-restrictions.md",
        "constraints": [],
    },
    "Cast": {
        "signature": "void Cast(LocalTensor<DstT>& dst, LocalTensor<SrcT>& src, int32_t count, uint8_t roundMode)",
        "dtypes": _DTYPES_ALL,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<DstT>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<SrcT>&", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
            {"name": "roundMode", "type": "uint8_t", "dir": "in"},
        ],
        "example": "Cast(dst, src, count, CAST_ROUND)  // float->half",
        "source": "api-precision.md",
        "constraints": [
            {"type": "alignment", "desc": "数据需 32 字节对齐", "severity": "warning"},
        ],
    },

    # ===== Data Movement =====
    "DataCopy": {
        "signature": "void DataCopy(LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)",
        "dtypes": _DTYPES_ALL,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "const GlobalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "uint32_t", "dir": "in"},
        ],
        "example": "DataCopy(localTensor, globalTensor, count)",
        "source": "api-datacopy.md",
        "constraints": [
            {"type": "alignment", "desc": "GM->UB 时不建议使用 DataCopy，改用 DataCopyPad", "severity": "error"},
        ],
    },
    "DataCopyPad": {
        "signature": "void DataCopyPad(LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count, uint32_t padSize)",
        "dtypes": _DTYPES_ALL,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "const GlobalTensor<T>&", "dir": "in"},
            {"name": "count", "type": "uint32_t", "dir": "in"},
            {"name": "padSize", "type": "uint32_t", "dir": "in"},
        ],
        "example": "DataCopyPad(localTensor, globalTensor, count, padBytes)",
        "source": "api-datacopy.md",
        "constraints": [],
    },

    # ===== Reduce =====
    "ReduceSum": {
        "signature": "void ReduceSum(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& workspace, int32_t count)",
        "dtypes": _DTYPES_FLOAT_HIGH,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "workspace", "type": "LocalTensor<T>&", "dir": "inout"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "ReduceSum(dst, src, tmpBuf, count)",
        "source": "api-reduce.md",
        "constraints": [
            {"type": "data_size", "desc": "dst 和 workspace 不能是同一 buffer", "severity": "error"},
        ],
    },
    "ReduceMax": {
        "signature": "void ReduceMax(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& workspace, int32_t count)",
        "dtypes": _DTYPES_FLOAT_HIGH,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<T>&", "dir": "out"},
            {"name": "src", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "workspace", "type": "LocalTensor<T>&", "dir": "inout"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "ReduceMax(dst, src, tmpBuf, count)",
        "source": "api-reduce.md",
        "constraints": [
            {"type": "data_size", "desc": "dst 和 workspace 不能是同一 buffer", "severity": "error"},
        ],
    },

    # ===== Compare =====
    "Compare": {
        "signature": "void Compare(LocalTensor<uint8_t>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, CMPMODE cmpMode, int32_t count)",
        "dtypes": _DTYPES_FLOAT,
        "repeat_limit": 255,
        "params": [
            {"name": "dst", "type": "LocalTensor<uint8_t>&", "dir": "out"},
            {"name": "src0", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "src1", "type": "LocalTensor<T>&", "dir": "in"},
            {"name": "cmpMode", "type": "CMPMODE", "dir": "in"},
            {"name": "count", "type": "int32_t", "dir": "in"},
        ],
        "example": "Compare(cmpLocal, srcLocal, maxLocal, CMPMODE::GT, A0_ALIGN)",
        "source": "api-restrictions.md",
        "constraints": [
            {"type": "alignment", "desc": "count 个元素所占空间必须 256 字节对齐", "severity": "error"},
        ],
    },
}

# API aliases / common name variations
_API_ALIASES: Dict[str, str] = {
    "GlobalTensor.SetValue": "DataCopyPad",
    "GlobalTensor.GetValue": "DataCopyPad",
    "SetValue": "DataCopyPad",
    "GetValue": "DataCopyPad",
}

# Blacklisted APIs with reasons and alternatives
_API_BLACKLIST: Dict[str, dict] = {
    "GlobalTensor::SetValue": {
        "reason": "效率极低，每次调用产生额外开销",
        "alternative": "DataCopyPad",
        "perf_impact": "高",
        "precision_impact": "无",
    },
    "GlobalTensor::GetValue": {
        "reason": "效率极低，每次调用产生额外开销",
        "alternative": "DataCopyPad",
        "perf_impact": "高",
        "precision_impact": "无",
    },
    "DataCopy(GM↔UB)": {
        "reason": "无法处理非对齐数据",
        "alternative": "DataCopyPad",
        "perf_impact": "中",
        "precision_impact": "无",
    },
}

# Alternative mappings: when primary API is not suitable
_API_ALTERNATIVES: Dict[str, List[dict]] = {
    "DataCopy": [
        {
            "api": "DataCopyPad",
            "steps": "将 DataCopy 替换为 DataCopyPad，添加 padSize 参数（非对齐时填充）",
            "performance_impact": "无显著影响",
            "precision_impact": "无",
        },
    ],
    "Sub": [
        {
            "api": "Adds(-scalar)",
            "steps": "当 src1 是标量时，使用 Adds(dst, src0, -scalar, count) 替代 Sub，减少 Duplicate 指令和 buffer 开销",
            "performance_impact": "正面：1 条指令 vs 2 条，32B buffer vs rLength*sizeof(T)",
            "precision_impact": "无",
        },
    ],
    "Div": [
        {
            "api": "Muls(1/scalar)",
            "steps": "当除数是标量时，预先计算倒数，使用 Muls 替代 Div",
            "performance_impact": "正面：减少除法开销",
            "precision_impact": "轻微：倒数计算可能引入舍入误差",
        },
    ],
}


# ============================================================
# Knowledge directory path
# ============================================================

def _get_knowledge_path() -> Optional[str]:
    """Get the Knowledge/api directory path."""
    this_dir = Path(__file__).parent
    knowledge = this_dir.parent / "Knowledge" / "api"
    if knowledge.is_dir():
        return str(knowledge)
    return None


def _read_doc(filename: str) -> str:
    """Read a documentation file from Knowledge/api/."""
    knowledge = _get_knowledge_path()
    if knowledge:
        filepath = os.path.join(knowledge, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                return f.read()
    return ""


# ============================================================
# Main retriever class
# ============================================================

class ApiDocRetriever:
    """
    API documentation retriever for Ascend C.

    Provides three capabilities sharing the same documentation source:
    1. lookup_signature()  — API signature lookup
    2. check_constraints() — API constraint validation
    3. find_alternatives() — API alternative finding
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        self._knowledge_path = knowledge_path or _get_knowledge_path()
        self._include_root = Path(self._knowledge_path) / "include" if self._knowledge_path else None

    def known_api_names(self) -> List[str]:
        names = set(_API_KNOWLEDGE.keys())
        names.update(_API_ALIASES.keys())
        names.update(_API_BLACKLIST.keys())
        names.update(_API_ALTERNATIVES.keys())
        return sorted(names)

    def is_available(self) -> bool:
        """Check if API docs are available."""
        return self._knowledge_path is not None

    @staticmethod
    def _normalize_api_name(api_name: str) -> str:
        name = (api_name or "").strip()
        for prefix in ("AscendC::", "ascendc::"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in _API_ALIASES:
            name = _API_ALIASES[name]
        return name

    def _iter_doc_paths(self) -> List[Path]:
        if not self._knowledge_path:
            return []
        return sorted(Path(self._knowledge_path).glob("*.md"))

    def _collect_doc_metadata(
        self,
        api_name: str,
        *,
        preferred_docs: Optional[List[str]] = None,
        limit: int = 4,
    ) -> List[dict]:
        if not self._knowledge_path:
            return []

        pattern = re.compile(rf"\b{re.escape(api_name)}\b", re.IGNORECASE)
        preferred = {item for item in (preferred_docs or []) if item}
        entries: List[dict] = []
        seen = set()

        def nearest_heading(lines: List[str], line_no: int) -> str:
            for index in range(line_no - 1, -1, -1):
                line = lines[index].strip()
                if line.startswith("#"):
                    return line.lstrip("#").strip()
            return ""

        doc_paths = self._iter_doc_paths()
        doc_paths.sort(key=lambda path: (path.name not in preferred, path.name))

        for path in doc_paths:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = text.splitlines()
            title = next((line.lstrip("#").strip() for line in lines if line.strip().startswith("#")), path.name)
            match_lines = [index + 1 for index, line in enumerate(lines) if pattern.search(line)]

            if not match_lines and path.name not in preferred:
                continue

            if not match_lines:
                key = (path.name, title)
                if key in seen:
                    continue
                seen.add(key)
                entries.append({
                    "path": path.name,
                    "title": title,
                    "section": "",
                    "excerpt": "",
                })
                if len(entries) >= limit:
                    break
                continue

            for line_no in match_lines[:2]:
                section = nearest_heading(lines, line_no)
                excerpt = lines[line_no - 1].strip()
                key = (path.name, section)
                if key in seen:
                    continue
                seen.add(key)
                entries.append({
                    "path": path.name,
                    "title": title,
                    "section": section,
                    "excerpt": excerpt[:240],
                })
                if len(entries) >= limit:
                    break
            if len(entries) >= limit:
                break
        return entries

    def _search_local_headers(self, api_name: str) -> _HeaderSearchResult:
        include_root = self._include_root
        if include_root is None or not include_root.is_dir():
            return _HeaderSearchResult()

        try:
            from .env_checker import _line_matches_api_symbol
        except Exception:
            _line_matches_api_symbol = None

        header_files = set()
        matches: List[str] = []
        fallback_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(api_name.split('::')[-1])}(?![A-Za-z0-9_])")

        for path in sorted(include_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            context = ""
            rel = path.relative_to(include_root).as_posix()
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith(("class ", "struct ", "enum ", "enum class ")):
                    context = stripped
                is_match = False
                if _line_matches_api_symbol is not None:
                    is_match = _line_matches_api_symbol(api_name, line, context=context)
                else:
                    is_match = bool(fallback_pattern.search(line))
                if not is_match:
                    continue
                header_files.add(rel)
                matches.append(f"{rel}:{line_no}:{stripped}")
                if len(matches) >= 16:
                    break
            if len(matches) >= 16:
                break

        signature = self._extract_header_signature(api_name, matches, include_root=include_root)
        summary = ""
        if header_files:
            summary = f"在本地 Knowledge/api/include 中找到 {len(header_files)} 个头文件匹配"
        return _HeaderSearchResult(
            header_files=sorted(header_files),
            matches=matches,
            signature=signature,
            summary=summary,
        )

    def lookup_signature(
        self,
        api_name: str,
        chip: str = "DAV_2201",
    ) -> ApiSignatureResult:
        """
        Look up API signature, supported dtypes, and repeatTimes limit.

        Args:
            api_name: API name (e.g., "DataCopy", "AscendC::Add")
            chip: Chip architecture

        Returns:
            ApiSignatureResult with signature details
        """
        del chip
        name = self._normalize_api_name(api_name)
        local_header_hit = self._search_local_headers(name)
        preferred_docs: List[str] = []

        if name in _API_KNOWLEDGE:
            info = _API_KNOWLEDGE[name]
            if info.get("source"):
                preferred_docs.append(info["source"])
            doc_metadata = self._collect_doc_metadata(name, preferred_docs=preferred_docs)
            details_lines = [
                f"API {name}: {info['signature']}",
                f"支持数据类型: {', '.join(info['dtypes'])}",
                f"repeatTimes 限制: {info.get('repeat_limit', 'N/A')}",
            ]
            if local_header_hit.header_files:
                details_lines.append("头文件: " + ", ".join(local_header_hit.header_files[:4]))
            if doc_metadata:
                details_lines.append("文档元数据: " + "; ".join(
                    f"{item['path']}|{item['section'] or item['title']}" for item in doc_metadata[:3]
                ))
            return ApiSignatureResult(
                api_name=name,
                signature=info["signature"],
                supported_dtypes=list(info["dtypes"]),
                repeat_times_limit=info.get("repeat_limit"),
                params=info.get("params", []),
                example_call=info.get("example", ""),
                source_doc=info.get("source", "unknown"),
                details="\n".join(details_lines),
                header_files=local_header_hit.header_files,
                doc_metadata=doc_metadata,
                match_kind="builtin_knowledge",
                confidence="high",
                is_actionable=bool(info["signature"]),
            )

        if local_header_hit.header_files:
            doc_metadata = self._collect_doc_metadata(name)
            details = [local_header_hit.summary]
            if local_header_hit.matches:
                details.append("匹配示例:")
                details.extend(local_header_hit.matches[:3])
            if doc_metadata:
                details.append("文档元数据: " + "; ".join(
                    f"{item['path']}|{item['section'] or item['title']}" for item in doc_metadata[:3]
                ))
            return ApiSignatureResult(
                api_name=name,
                signature=local_header_hit.signature,
                supported_dtypes=[],
                repeat_times_limit=None,
                params=[],
                example_call="",
                source_doc=", ".join(local_header_hit.header_files[:3]),
                details="\n".join(details),
                header_files=local_header_hit.header_files,
                doc_metadata=doc_metadata,
                match_kind="header_decl" if local_header_hit.signature else "doc_excerpt",
                confidence="high" if local_header_hit.signature else "low",
                is_actionable=bool(local_header_hit.signature),
            )

        doc_hit = self._search_docs(name)
        if doc_hit is not None:
            return doc_hit

        header_hit = self._search_headers(name)
        if header_hit is not None:
            return header_hit

        return ApiSignatureResult(
            api_name=api_name,
            signature="",
            supported_dtypes=[],
            repeat_times_limit=None,
            params=[],
            example_call="",
            source_doc="",
            details=f"未找到 API '{api_name}' 的签名信息。\n"
                    f"可能原因: API 名称不正确、或该 API 未在知识库中收录。\n"
                    f"建议: 查阅 asc-devkit/docs/api/context/ 下的官方 API 文档。",
            header_files=[],
            doc_metadata=[],
                match_kind="not_found",
                confidence="low",
                is_actionable=False,
        )

    def _search_docs(self, api_name: str) -> Optional[ApiSignatureResult]:
        if not self._knowledge_path:
            return None
        doc_metadata = self._collect_doc_metadata(api_name)
        if doc_metadata:
            primary = doc_metadata[0]
            details = f"在文档 {primary['path']} 中检索到 {api_name} 的相关信息"
            if primary.get("excerpt"):
                details += f": {primary['excerpt']}"
            return ApiSignatureResult(
                api_name=api_name,
                signature="",
                supported_dtypes=[],
                repeat_times_limit=None,
                params=[],
                example_call="",
                source_doc=primary["path"],
                details=details,
                header_files=[],
                doc_metadata=doc_metadata,
                match_kind="doc_excerpt",
                confidence="low",
                is_actionable=False,
            )
        return None

    def _search_headers(self, api_name: str) -> Optional[ApiSignatureResult]:
        try:
            from .env_checker import check_api_exists
        except Exception:
            return None

        header_result = check_api_exists(api_name)
        if not header_result.found:
            return None

        signature = self._extract_header_signature(api_name, header_result.matches)
        if not signature:
            return None

        details = [header_result.summary]
        if header_result.matches:
            details.append("匹配示例:")
            details.extend(header_result.matches[:3])

        return ApiSignatureResult(
            api_name=api_name,
            signature=signature,
            supported_dtypes=[],
            repeat_times_limit=None,
            params=[],
            example_call="",
            source_doc=", ".join(header_result.header_files[:3]),
            details="\n".join(details),
            header_files=header_result.header_files,
            doc_metadata=self._collect_doc_metadata(api_name),
            match_kind="header_decl",
            confidence="medium",
            is_actionable=True,
        )

    @staticmethod
    def _extract_header_signature(api_name: str, matches: List[str], include_root: Optional[Path] = None) -> str:
        parts = [part for part in re.split(r"\s*::\s*", (api_name or "").strip()) if part]
        if parts and parts[0].lower() == "ascendc":
            parts = parts[1:]
        if not parts:
            return ""
        short = re.escape(parts[-1])

        type_decl_re = re.compile(
            rf"\b(?:using|typedef|class|struct|enum(?:\s+class)?)\s+[^;={{}}]*\b{short}\b"
        )
        macro_re = re.compile(rf"^\s*#\s*define\s+{short}\b")
        tail_call_re = re.compile(rf"(?<![A-Za-z0-9_]){short}(?![A-Za-z0-9_])\s*(?:<|\()")
        decl_cue_re = re.compile(
            r"\b(?:inline|constexpr|static|extern|virtual|friend|template|__aicore__|__host__|__device__|void|bool|char|short|int|long|float|double|size_t|uint\d+_t|int\d+_t)\b"
        )

        def read_declaration(path_str: str, line_no: int) -> str:
            if include_root is None:
                return ""
            try:
                path = include_root / path_str
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                return ""
            start = max(line_no - 1, 0)
            collected: List[str] = []
            found_open = False
            for line in lines[start:start + 8]:
                if _is_non_signature_line(line):
                    if collected:
                        break
                    continue
                stripped = line.strip()
                if not collected and short.lower() not in stripped.lower():
                    continue
                collected.append(stripped)
                if "(" in stripped:
                    found_open = True
                if found_open and any(token in stripped for token in (")", ";", "{")):
                    break
            text = " ".join(collected)
            if "(" not in text:
                return ""
            return re.sub(r"\s+", " ", text).strip()

        for match in matches:
            parts_match = match.split(":", 2)
            line = parts_match[2].strip() if ":" in match else match.strip()
            code = line.split("//", 1)[0].strip()
            if not code:
                continue
            if _is_non_signature_line(code):
                continue
            if type_decl_re.search(code) or macro_re.search(code):
                return code
            token_match = tail_call_re.search(code)
            if not token_match:
                if include_root is not None and len(parts_match) >= 3 and parts_match[1].isdigit():
                    multi_line_decl = read_declaration(parts_match[0], int(parts_match[1]))
                    if multi_line_decl:
                        return multi_line_decl
                continue
            prefix = code[:token_match.start()].rstrip()
            if prefix.endswith("::"):
                qualifier = prefix[:-2].rstrip()
                if decl_cue_re.search(qualifier) or " " in qualifier:
                    if include_root is not None and len(parts_match) >= 3 and parts_match[1].isdigit():
                        return read_declaration(parts_match[0], int(parts_match[1])) or code
                    return code
                continue
            if decl_cue_re.search(prefix) or re.search(r"[&*>]\s*$", prefix):
                if include_root is not None and len(parts_match) >= 3 and parts_match[1].isdigit():
                    return read_declaration(parts_match[0], int(parts_match[1])) or code
                return code
        return ""

    def check_constraints(
        self,
        api_name: str,
        call_context: Dict,
    ) -> ApiConstraintResult:
        """
        Check if a call context violates API constraints.

        Args:
            api_name: API name
            call_context: Dict with call parameters, e.g.:
                {
                    "count": 256,
                    "dtype": "float",
                    "chip": "DAV_2201",
                    "repeat_times": 256,
                    "is_gm_to_ub": True,
                }

        Returns:
            ApiConstraintResult with constraint violations and suggestions
        """
        name = api_name.strip()
        for prefix in ("AscendC::", "ascendc::"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in _API_ALIASES:
            name = _API_ALIASES[name]

        constraints: List[dict] = []
        violations: List[str] = []
        suggestions: List[str] = []
        checks_performed: List[dict] = []
        unknowns: List[str] = []
        checked_context = {
            key: value
            for key, value in (call_context or {}).items()
            if value is not None and value != ""
        }
        source_doc = ""
        missing_required_context = False

        if name in _API_KNOWLEDGE:
            constraints = list(_API_KNOWLEDGE[name].get("constraints", []))
            source_doc = _API_KNOWLEDGE[name].get("source", "")

        def require_context(*keys: str) -> bool:
            nonlocal missing_required_context
            missing = [key for key in keys if key not in checked_context]
            if missing:
                missing_required_context = True
                for key in missing:
                    unknowns.append(f"未提供 {key}，无法完成该项约束校验。")
                return False
            return True

        # === Universal constraint checks ===

        # Check 1: repeatTimes <= 255
        if "repeat_times" in checked_context:
            repeat_times = int(checked_context.get("repeat_times", 0))
            if repeat_times > 255:
                constraints.append({
                    "type": "repeat_times",
                    "desc": "repeatTimes 为 uint8_t 时最大值 255",
                    "severity": "error",
                })
                violations.append(
                    f"repeat_times = {repeat_times}，超过 uint8_t 最大值 255。"
                    f"会导致溢出为 {repeat_times % 256}，计算结果错误。"
                )
                suggestions.append(
                    "分批处理: 使用 while 循环，每次最多处理 255 个 repeat。"
                    "参考 api-repeat-limits.md 中的方案二。"
                )
                checks_performed.append({
                    "name": "repeat_times_limit",
                    "status": "fail",
                    "detail": f"repeat_times={repeat_times}，超过 255。",
                })
            else:
                checks_performed.append({
                    "name": "repeat_times_limit",
                    "status": "pass",
                    "detail": f"repeat_times={repeat_times}，未超过 255。",
                })
        else:
            unknowns.append("未提供 repeat_times，未校验 repeatTimes 的 uint8_t 上限。")

        # Check 2: GM->UB alignment
        if name == "DataCopy":
            if "is_gm_to_ub" in checked_context:
                is_gm_to_ub = bool(checked_context.get("is_gm_to_ub", False))
                if is_gm_to_ub:
                    if require_context("count", "dtype"):
                        dtype_bytes_map = {"float": 4, "float32": 4, "half": 2, "float16": 2,
                                           "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bfloat16": 2}
                        dtype_name = str(checked_context.get("dtype", "")).lower()
                        elem_size = dtype_bytes_map.get(dtype_name)
                        if elem_size is None:
                            missing_required_context = True
                            unknowns.append(f"dtype={checked_context.get('dtype')} 未知，无法判断对齐。")
                        else:
                            total_bytes = int(checked_context["count"]) * elem_size
                            if total_bytes % 32 != 0:
                                constraints.append({
                                    "type": "alignment",
                                    "desc": "GM->UB 非 32B 对齐搬运应使用 DataCopyPad",
                                    "severity": "error",
                                })
                                violations.append(
                                    f"DataCopy(GM->UB) 搬运长度 {total_bytes}B 不是 32B 对齐。"
                                )
                                suggestions.append("使用 DataCopyPad 或 padding 对齐到 32B。")
                                checks_performed.append({
                                    "name": "gm_to_ub_alignment",
                                    "status": "fail",
                                    "detail": f"总字节数 {total_bytes}B，不满足 32B 对齐。",
                                })
                            else:
                                checks_performed.append({
                                    "name": "gm_to_ub_alignment",
                                    "status": "pass",
                                    "detail": f"总字节数 {total_bytes}B，满足 32B 对齐。",
                                })
                else:
                    checks_performed.append({
                        "name": "gm_to_ub_alignment",
                        "status": "pass",
                        "detail": "当前上下文不是 GM->UB，未触发 DataCopyPad 替代规则。",
                    })
            else:
                unknowns.append("未提供 is_gm_to_ub，无法判断 DataCopy 是否处于 GM->UB 场景。")
                missing_required_context = True

        if name == "DataCopyPad":
            if require_context("count", "dtype", "pad_size"):
                checks_performed.append({
                    "name": "datacopy_pad_context",
                    "status": "pass",
                    "detail": "已提供 count、dtype、pad_size，可进一步推导 padding 合法性。",
                })

        if name == "ReduceSum":
            have_alias = require_context("workspace_alias")
            have_workspace_size = require_context("workspace_size_bytes")
            if have_alias:
                if bool(checked_context.get("workspace_alias")):
                    violations.append("ReduceSum 的 dst 和 workspace 不能复用同一 buffer。")
                    suggestions.append("为 ReduceSum 分配独立 workspace buffer。")
                    checks_performed.append({
                        "name": "reduce_workspace_alias",
                        "status": "fail",
                        "detail": "workspace_alias=True。",
                    })
                else:
                    checks_performed.append({
                        "name": "reduce_workspace_alias",
                        "status": "pass",
                        "detail": "workspace_alias=False。",
                    })
            if have_workspace_size:
                checks_performed.append({
                    "name": "reduce_workspace_size",
                    "status": "pass",
                    "detail": f"workspace_size_bytes={checked_context.get('workspace_size_bytes')}",
                })

        if name == "Duplicate":
            have_count = require_context("count")
            have_repeat_times = require_context("repeat_times")
            if have_count and have_repeat_times:
                checks_performed.append({
                    "name": "duplicate_repeat_context",
                    "status": "pass",
                    "detail": f"count={checked_context.get('count')}, repeat_times={checked_context.get('repeat_times')}",
                })
            else:
                require_context("mask")

        if name == "Cast":
            have_cast_dtypes = require_context("src_dtype", "dst_dtype")
            if have_cast_dtypes:
                supported = {item.lower() for item in _DTYPES_ALL}
                src_dtype = str(checked_context.get("src_dtype", "")).lower()
                dst_dtype = str(checked_context.get("dst_dtype", "")).lower()
                if src_dtype not in supported or dst_dtype not in supported:
                    violations.append(f"Cast 的 dtype 组合非法: {src_dtype} -> {dst_dtype}。")
                    suggestions.append("改用受支持的 Cast 源/目标 dtype 组合。")
                    checks_performed.append({
                        "name": "cast_dtype_pair",
                        "status": "fail",
                        "detail": f"src_dtype={src_dtype}, dst_dtype={dst_dtype}",
                    })
                else:
                    checks_performed.append({
                        "name": "cast_dtype_pair",
                        "status": "pass",
                        "detail": f"src_dtype={src_dtype}, dst_dtype={dst_dtype}",
                    })

        # Check 3: Compare API 256B alignment
        count = checked_context.get("count")
        dtype = str(checked_context.get("dtype", "")).lower()
        if name == "Compare":
            if count is None:
                unknowns.append("未提供 count，未校验 Compare 的 256B 对齐约束。")
            if not dtype:
                unknowns.append("未提供 dtype，未校验 Compare 的元素字节数和 256B 对齐约束。")
            if count is not None and dtype:
                dtype_bytes_map = {"float": 4, "float32": 4, "half": 2, "float16": 2,
                                   "int32": 4, "int16": 2, "int8": 1, "uint8": 1}
                elem_size = dtype_bytes_map.get(dtype, 4)
                total_bytes = int(count) * elem_size
                if total_bytes % 256 != 0:
                    constraints.append({
                        "type": "alignment",
                        "desc": "Compare API 的 count 个元素所占空间必须 256 字节对齐",
                        "severity": "error",
                    })
                    violations.append(
                        f"count={count} * sizeof({dtype})={elem_size} = {total_bytes}B，"
                        f"不是 256B 的倍数。"
                    )
                    aligned_count = ((int(count) + (256 // elem_size - 1)) // (256 // elem_size)) * (256 // elem_size)
                    suggestions.append(
                        f"使用 padding 策略: 将 count 从 {count} 对齐到 {aligned_count}，"
                        f"多余的元素填充极值。"
                    )
                    checks_performed.append({
                        "name": "compare_256b_alignment",
                        "status": "fail",
                        "detail": f"总字节数 {total_bytes}B，不是 256B 的倍数。",
                    })
                else:
                    checks_performed.append({
                        "name": "compare_256b_alignment",
                        "status": "pass",
                        "detail": f"总字节数 {total_bytes}B，满足 256B 对齐。",
                    })

        # Check 4: UB capacity
        if "ub_usage_bytes" in checked_context:
            ub_usage = int(checked_context.get("ub_usage_bytes", 0))
            ub_capacity = int(checked_context.get("ub_capacity_bytes", 196608))
            if ub_usage > ub_capacity:
                constraints.append({
                    "type": "data_size",
                    "desc": "UB Buffer 使用量不能超过 UB 容量",
                    "severity": "error",
                })
                violations.append(
                    f"UB 使用 {ub_usage}B 超过容量 {ub_capacity}B"
                )
                suggestions.append("减小 tile_size 或使用分批处理。")
                checks_performed.append({
                    "name": "ub_capacity",
                    "status": "fail",
                    "detail": f"UB 使用 {ub_usage}B，超过容量 {ub_capacity}B。",
                })
            else:
                checks_performed.append({
                    "name": "ub_capacity",
                    "status": "pass",
                    "detail": f"UB 使用 {ub_usage}B，未超过容量 {ub_capacity}B。",
                })
        else:
            unknowns.append("未提供 ub_usage_bytes，未校验 UB 容量是否溢出。")

        if violations:
            compliance_status = "fail"
        elif missing_required_context:
            compliance_status = "insufficient_context"
        else:
            compliance_status = "pass"

        is_compliant = compliance_status == "pass"
        if suggestions:
            suggestion_text = "\n".join(suggestions)
        elif compliance_status == "insufficient_context":
            suggestion_text = "缺少关键上下文，建议补充结构化参数后重试。"
        elif unknowns:
            suggestion_text = "未发现明确违规，但仍有未校验项。"
        else:
            suggestion_text = "当前调用符合已知约束。"

        return ApiConstraintResult(
            api_name=api_name,
            constraints=constraints,
            violations=violations,
            suggestion=suggestion_text,
            is_compliant=is_compliant,
            checked_context=checked_context,
            checks_performed=checks_performed,
            unknowns=unknowns,
            source_doc=source_doc,
            compliance_status=compliance_status,
        )

    def find_alternatives(
        self,
        api_name: str,
        reason: str = "不可用",
    ) -> ApiAlternativeResult:
        """
        Find alternative APIs when the primary API is not suitable.

        Args:
            api_name: Primary API name
            reason: Why the primary API is not suitable
                    ("不存在" / "性能差" / "精度差" / other)

        Returns:
            ApiAlternativeResult with alternatives
        """
        name = api_name.strip()
        for prefix in ("AscendC::", "ascendc::"):
            if name.startswith(prefix):
                name = name[len(prefix):]

        alternatives: List[dict] = []

        # Check blacklist
        for bl_name, bl_info in _API_BLACKLIST.items():
            bl_short = bl_name.split("::")[-1].split("(")[0]
            if name == bl_name or name == bl_short:
                alternatives.append({
                    "api": bl_info["alternative"],
                    "steps": f"使用 {bl_info['alternative']} 替代 {bl_name}",
                    "performance_impact": bl_info["perf_impact"],
                    "precision_impact": bl_info["precision_impact"],
                })

        # Check alternatives map
        if name in _API_ALTERNATIVES:
            for alt in _API_ALTERNATIVES[name]:
                # Avoid duplicates
                if alt["api"] not in [a["api"] for a in alternatives]:
                    alternatives.append(alt)

        # Check if there's a known blacklist reason
        bl_reason = None
        for bl_name, bl_info in _API_BLACKLIST.items():
            bl_short = bl_name.split("::")[-1].split("(")[0]
            if name == bl_name or name == bl_short:
                bl_reason = bl_info["reason"]
                break

        if alternatives:
            recommended = alternatives[0]["api"]
        else:
            recommended = "未找到已知替代方案，建议查阅官方文档"

        return ApiAlternativeResult(
            primary_api=api_name,
            alternatives=alternatives,
            recommended=recommended,
            reason=bl_reason or reason,
        )

    def list_known_apis(self) -> List[str]:
        """List all known API names in the knowledge base."""
        return sorted(_API_KNOWLEDGE.keys())
