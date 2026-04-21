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


@dataclass
class ApiConstraintResult:
    """Result of API constraint check."""
    api_name: str
    constraints: List[dict]                 # [{type, description, severity}]
    violations: List[str]                   # Current violations
    suggestion: str                         # Fix suggestion
    is_compliant: bool                      # Whether current call is compliant


@dataclass
class ApiAlternativeResult:
    """Result of API alternative lookup."""
    primary_api: str
    alternatives: List[dict]                # [{api, steps, performance_impact, precision_impact}]
    recommended: str                        # Recommended approach
    reason: str                             # Why alternatives are needed


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

    def __init__(self):
        self._knowledge_path = _get_knowledge_path()

    def known_api_names(self) -> List[str]:
        names = set(_API_KNOWLEDGE.keys())
        names.update(_API_ALIASES.keys())
        names.update(_API_BLACKLIST.keys())
        names.update(_API_ALTERNATIVES.keys())
        return sorted(names)

    def is_available(self) -> bool:
        """Check if API docs are available."""
        return self._knowledge_path is not None

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
        # Normalize: strip AscendC:: prefix
        name = api_name.strip()
        for prefix in ("AscendC::", "ascendc::"):
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Try alias
        if name in _API_ALIASES:
            name = _API_ALIASES[name]

        if name in _API_KNOWLEDGE:
            info = _API_KNOWLEDGE[name]
            return ApiSignatureResult(
                api_name=name,
                signature=info["signature"],
                supported_dtypes=list(info["dtypes"]),
                repeat_times_limit=info.get("repeat_limit"),
                params=info.get("params", []),
                example_call=info.get("example", ""),
                source_doc=info.get("source", "unknown"),
                details=f"API {name}: {info['signature']}\n"
                        f"支持数据类型: {', '.join(info['dtypes'])}\n"
                        f"repeatTimes 限制: {info.get('repeat_limit', 'N/A')}",
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
        )

    def _search_docs(self, api_name: str) -> Optional[ApiSignatureResult]:
        if not self._knowledge_path:
            return None
        pattern = re.compile(rf"\b{re.escape(api_name)}\b", re.IGNORECASE)
        for path in sorted(Path(self._knowledge_path).glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            match = pattern.search(text)
            if not match:
                continue
            start = max(0, match.start() - 160)
            end = min(len(text), match.end() + 240)
            snippet = " ".join(text[start:end].split())
            return ApiSignatureResult(
                api_name=api_name,
                signature="",
                supported_dtypes=[],
                repeat_times_limit=None,
                params=[],
                example_call="",
                source_doc=path.name,
                details=f"在文档 {path.name} 中检索到 {api_name} 的相关片段: {snippet}",
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

        signature = ""
        if header_result.matches:
            parts = header_result.matches[0].split(":", 2)
            signature = parts[2].strip() if len(parts) >= 3 else header_result.matches[0].strip()

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
        )

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

        if name in _API_KNOWLEDGE:
            constraints = list(_API_KNOWLEDGE[name].get("constraints", []))

        # === Universal constraint checks ===

        # Check 1: repeatTimes <= 255
        repeat_times = call_context.get("repeat_times", 0)
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

        # Check 2: GM->UB alignment
        is_gm_to_ub = call_context.get("is_gm_to_ub", False)
        if is_gm_to_ub and name == "DataCopy":
            constraints.append({
                "type": "alignment",
                "desc": "GM->UB 搬运应使用 DataCopyPad 而非 DataCopy",
                "severity": "error",
            })
            violations.append(
                "DataCopy(GM->UB) 无法处理非对齐数据。"
            )
            suggestions.append("使用 DataCopyPad 替代 DataCopy 进行 GM->UB 数据搬运。")

        # Check 3: Compare API 256B alignment
        count = call_context.get("count", 0)
        dtype = call_context.get("dtype", "float")
        if name == "Compare" and count > 0:
            dtype_bytes_map = {"float": 4, "float32": 4, "half": 2, "float16": 2,
                               "int32": 4, "int16": 2, "int8": 1, "uint8": 1}
            elem_size = dtype_bytes_map.get(dtype, 4)
            total_bytes = count * elem_size
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
                aligned_count = ((count + (256 // elem_size - 1)) // (256 // elem_size)) * (256 // elem_size)
                suggestions.append(
                    f"使用 padding 策略: 将 count 从 {count} 对齐到 {aligned_count}，"
                    f"多余的元素填充极值。"
                )

        # Check 4: UB capacity
        ub_usage = call_context.get("ub_usage_bytes", 0)
        ub_capacity = call_context.get("ub_capacity_bytes", 196608)
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

        is_compliant = len(violations) == 0
        suggestion_text = "\n".join(suggestions) if suggestions else "当前调用符合约束。"

        return ApiConstraintResult(
            api_name=api_name,
            constraints=constraints,
            violations=violations,
            suggestion=suggestion_text,
            is_compliant=is_compliant,
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
