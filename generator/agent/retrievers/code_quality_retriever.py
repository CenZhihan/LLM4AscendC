"""
Code Quality Retriever for Ascend C kernel development agent.

Provides coding style check and security pattern check based on
Knowledge/code-review/ documentation files.

Two capabilities share a single retriever because they both scan
code against rule files:
1. check_style()   — Ascend C coding convention check
2. check_security() — Security pattern detection
"""
import os
import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Structured result types
# ============================================================

@dataclass
class CodingStyleResult:
    """Result of coding style check."""
    passed: bool
    issues: List[dict]              # [{line, severity, rule, message, suggestion}]
    score: int                      # 0-100
    # rule types: "naming" | "structure" | "api_usage" | "memory" | "sync"


@dataclass
class SecurityCheckResult:
    """Result of security pattern check."""
    safe: bool
    issues: List[dict]              # [{type, location, severity, description, fix}]
    # type: "buffer_overflow" | "null_pointer" | "division_by_zero" | "memory_leak" | ...


# ============================================================
# Coding style rules
# ============================================================

@dataclass
class StyleRule:
    """A single coding style rule."""
    rule_id: str
    category: str                   # "naming" | "structure" | "api_usage" | "memory" | "sync"
    pattern: str                    # Regex pattern to detect violation
    message: str
    suggestion: str
    severity: str                   # "error" | "warning" | "info"


# Rules extracted from Knowledge/code-review/C++CodeStyle.md
# and Knowledge/code-review/C++GeneralCoding.md
_STYLE_RULES: List[StyleRule] = [
    # Naming rules
    StyleRule(
        rule_id="NAME-001",
        category="naming",
        pattern=r"\b[a-z][a-z0-9]*_[A-Z]",
        message="命名风格不一致：混用 snake_case 和 UPPER_CASE",
        suggestion="Ascend C 项目使用 PascalCase 类名，camelCase 函数/变量名",
        severity="warning",
    ),
    StyleRule(
        rule_id="NAME-002",
        category="naming",
        pattern=r"#include\s*[\"<][a-z_]+\.hpp[\">]",
        message="头文件使用 .hpp 扩展名，Ascend C 项目统一使用 .h",
        suggestion="将 .hpp 改为 .h",
        severity="warning",
    ),

    # Structure rules
    StyleRule(
        rule_id="STRUCT-001",
        category="structure",
        pattern=r"^\s*#include\s*[\"<](algorithm|cmath|vector|map|set|unordered_map|unordered_set)[\">]",
        message="Kernel 侧代码禁止包含 C++ 标准库头文件",
        suggestion="使用 Ascend C 提供的专用 API 替代 std:: 函数",
        severity="error",
    ),
    StyleRule(
        rule_id="STRUCT-002",
        category="structure",
        pattern=r"\bnew\s+",
        message="AI Core 不支持动态内存分配",
        suggestion="使用静态分配: constexpr uint32_t SIZE 或 pipe.InitBuffer()",
        severity="error",
    ),
    StyleRule(
        rule_id="STRUCT-003",
        category="structure",
        pattern=r"\bmalloc\s*\(",
        message="AI Core 不支持动态内存分配",
        suggestion="使用静态分配或 TBuf/TQue",
        severity="error",
    ),
    StyleRule(
        rule_id="STRUCT-004",
        category="structure",
        pattern=r"\bstd::vector\b",
        message="AI Core 不支持 std::vector（动态分配）",
        suggestion="使用静态数组或 Ascend C TBuf",
        severity="error",
    ),

    # API usage rules
    StyleRule(
        rule_id="API-001",
        category="api_usage",
        pattern=r"\bstd::(abs|min|max|sqrt|pow|exp|log|log2|log10|sin|cos|tan|floor|ceil|round)\s*\(",
        message="Kernel 侧禁止使用 std:: 数学函数",
        suggestion="使用 AscendC::Abs/Min/Max/Sqrt/Exp/Log 等 API",
        severity="error",
    ),
    StyleRule(
        rule_id="API-002",
        category="api_usage",
        pattern=r"\bstd::isnan\s*\(|\bstd::isinf\s*\(",
        message="std::isnan/isinf 在 Kernel 侧不可用",
        suggestion="手动实现特殊值检查逻辑",
        severity="error",
    ),
    StyleRule(
        rule_id="API-003",
        category="api_usage",
        pattern=r"GlobalTensor::\w*Value\s*\(",
        message="GlobalTensor::SetValue/GetValue 效率极低",
        suggestion="使用 DataCopyPad 进行数据搬运",
        severity="warning",
    ),
    StyleRule(
        rule_id="API-004",
        category="api_usage",
        pattern=r"const\s+uint(8|16|32|64)_t\s+\w+\s*=\s*",
        message="Buffer 大小、循环次数等推荐使用 constexpr",
        suggestion="将 const 改为 constexpr 以允许编译期优化",
        severity="info",
    ),

    # Memory rules
    StyleRule(
        rule_id="MEM-001",
        category="memory",
        pattern=r"DeleteTensor\s*\(",
        message="Ascend C 中不建议手动释放 Tensor",
        suggestion="依赖 UB 生命周期管理，避免手动 DeleteTensor",
        severity="warning",
    ),

    # Sync rules
    StyleRule(
        rule_id="SYNC-001",
        category="sync",
        pattern=r"AscendC::printf\s*\(.*\)\s*;",
        message="AscendC::printf 仅用于调试，不建议在生产代码中使用",
        suggestion="移除调试打印或使用 log 机制",
        severity="info",
    ),
]

# Security patterns
# Extracted from Knowledge/code-review/C++SecureCoding.md
_SECURITY_PATTERNS: List[Tuple[str, str, str, str, str]] = [
    # (regex, type, severity, description, fix)
    (r"\bdelete\s+", "memory_leak", "error",
     "AI Core 不支持 delete，可能导致未定义行为",
     "使用 UB 生命周期管理，不需要手动释放"),

    (r"\bfree\s*\(", "memory_leak", "error",
     "AI Core 不支持 free",
     "使用静态分配或 TBuf/TQue 管理内存"),

    (r"\[.*\]\s*(?!\s*=\s*\{)", "buffer_overflow", "warning",
     "数组下标访问未进行边界检查",
     "确保下标在有效范围内 [0, size)"),

    (r"/\s*(?!/)", "division_by_zero", "warning",
     "除法操作未检查除数是否为零",
     "在除法前添加除数非零检查"),

    (r"reinterpret_cast\s*<", "type_safety", "warning",
     "reinterpret_cast 存在类型安全隐患",
     "使用 static_cast 或避免类型转换"),

    (r"goto\s+", "control_flow", "info",
     "使用 goto 降低代码可读性",
     "使用结构化控制流替代 goto"),

    (r"volatile\s+", "concurrency", "warning",
     "volatile 不能保证线程安全",
     "使用 Ascend C 提供的同步原语"),

    (r"memcpy\s*\(", "buffer_overflow", "warning",
     "memcpy 不进行边界检查",
     "确保目标 buffer 足够大，或使用 DataCopy"),

    (r"memset\s*\(", "buffer_overflow", "warning",
     "memset 不进行边界检查",
     "确保目标 buffer 足够大"),

    (r"printf\s*\(", "security", "info",
     "使用标准 printf 而非 AscendC::printf",
     "在 Kernel 侧使用 AscendC::printf"),
]


# ============================================================
# Knowledge directory path
# ============================================================

def _get_knowledge_path() -> Optional[str]:
    """Get the Knowledge/code-review directory path."""
    this_dir = Path(__file__).parent
    knowledge = this_dir.parent / "Knowledge" / "code-review"
    if knowledge.is_dir():
        return str(knowledge)
    return None


# ============================================================
# Code quality checker
# ============================================================

def check_coding_style(
    code: str,
    ruleset: str = "ascendc_v2",
) -> CodingStyleResult:
    """
    Check code against Ascend C coding conventions.

    Args:
        code: Source code to check
        ruleset: Ruleset identifier (currently only "ascendc_v2" is supported)

    Returns:
        CodingStyleResult with issues and score
    """
    issues: List[dict] = []
    lines = code.splitlines()

    for rule in _STYLE_RULES:
        for line_num, line in enumerate(lines, 1):
            if re.search(rule.pattern, line):
                issues.append({
                    "line": line_num,
                    "severity": rule.severity,
                    "rule": rule.rule_id,
                    "category": rule.category,
                    "message": rule.message,
                    "suggestion": rule.suggestion,
                    "code_snippet": line.strip()[:120],
                })

    # Calculate score (100 = perfect, deduct for issues)
    score = 100
    for issue in issues:
        if issue["severity"] == "error":
            score -= 10
        elif issue["severity"] == "warning":
            score -= 5
        elif issue["severity"] == "info":
            score -= 1
    score = max(0, score)

    # Deduplicate issues (same rule, same line)
    seen = set()
    unique_issues = []
    for issue in issues:
        key = (issue["line"], issue["rule"])
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    passed = not any(i["severity"] == "error" for i in unique_issues)

    return CodingStyleResult(
        passed=passed,
        issues=unique_issues,
        score=score,
    )


def check_security_patterns(
    code: str,
    op_type: str = "elementwise",
) -> SecurityCheckResult:
    """
    Check code for security issues.

    Args:
        code: Source code to check
        op_type: Operator type (for context-specific checks)

    Returns:
        SecurityCheckResult with security issues
    """
    issues: List[dict] = []
    lines = code.splitlines()

    for pattern, issue_type, severity, description, fix in _SECURITY_PATTERNS:
        for line_num, line in enumerate(lines, 1):
            if re.search(pattern, line):
                issues.append({
                    "type": issue_type,
                    "location": f"line {line_num}",
                    "severity": severity,
                    "description": description,
                    "fix": fix,
                    "code_snippet": line.strip()[:120],
                })

    # Additional checks for specific operator types
    if op_type == "reduce":
        # Check for dst == workspace in reduce operations
        if re.search(r"Reduce\w+.*dst.*dst", code):
            issues.append({
                "type": "buffer_alias",
                "location": "reduce call",
                "severity": "error",
                "description": "Reduce API 的 dst 和 workspace 不能是同一 buffer",
                "fix": "使用不同的 LocalTensor 作为 dst 和 workspace",
                "code_snippet": "",
            })

    safe = not any(i["severity"] == "error" for i in issues)

    return SecurityCheckResult(
        safe=safe,
        issues=issues,
    )


# ============================================================
# Retriever class
# ============================================================

class CodeQualityRetriever:
    """
    Code quality checker for Ascend C.

    Provides two capabilities sharing the same rule files:
    1. check_style()   — Coding convention check
    2. check_security() — Security pattern detection
    """

    def __init__(self):
        self._knowledge_path = _get_knowledge_path()

    def is_available(self) -> bool:
        """Code quality check is always available (rules are built-in)."""
        return True

    def check_style(
        self,
        code: str,
        ruleset: str = "ascendc_v2",
    ) -> CodingStyleResult:
        """
        Check code against Ascend C coding conventions.

        Args:
            code: Source code to check
            ruleset: Ruleset identifier

        Returns:
            CodingStyleResult with issues and score
        """
        return check_coding_style(code, ruleset)

    def check_security(
        self,
        code: str,
        op_type: str = "elementwise",
    ) -> SecurityCheckResult:
        """
        Check code for security issues.

        Args:
            code: Source code to check
            op_type: Operator type for context

        Returns:
            SecurityCheckResult with security issues
        """
        return check_security_patterns(code, op_type)
