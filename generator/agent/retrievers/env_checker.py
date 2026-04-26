"""
Environment Checker for Ascend C kernel development agent.

Provides three structured tools:
1. check_env()       — CANN environment overview (version, tools, OPP packages)
2. query_npu_devices() — NPU device information via npu-smi
3. check_api_exists()  — API compatibility check in CANN headers

Adapts the CANN skill's environment checking capability (scripts/check_env.sh,
scripts/npu_info.sh) into Python functions with structured return types.
"""
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


# Default CANN paths
_CANN_TOOLKIT_CANDIDATES = [
    "/usr/local/Ascend/ascend-toolkit/latest",
    os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
]

# NPU query type -> npu-smi command mapping
_NPU_QUERY_MAP = {
    "info": "npu-smi info",
    "list": "npu-smi list",
    "memory": "npu-smi info -t memory",
    "temp": "npu-smi info -t temp",
    "power": "npu-smi info -t power",
    "usages": "npu-smi info -t usages",
}


# ============================================================
# Structured result types
# ============================================================

@dataclass
class EnvCheckResult:
    """Result of CANN environment overview check."""
    all_passed: bool
    cann_version: str
    cann_home: Optional[str]
    opp_path: Optional[str]
    ld_library_path: str
    tools: Dict[str, bool]          # {"msprof": True, "cannsim": False, "npu-smi": True}
    custom_opps: List[str]          # installed vendor names
    errors: int
    warnings: int
    details: str                    # human-readable summary


@dataclass
class NpuDeviceResult:
    """Result of NPU device query."""
    available: bool
    query_type: str                 # "info" / "list" / "memory" / ...
    raw_output: str


@dataclass
class ApiCheckResult:
    """Result of API compatibility check."""
    found: bool
    api_name: str
    header_files: List[str]         # matching header file paths
    matches: List[str]              # first 5 matching lines
    summary: str                    # human-readable summary


_NPU_INFO_SCOPED_TYPES = {"memory", "temp", "power", "usages"}


def _parse_npu_mapping(raw_output: str) -> Dict[int, int]:
    """Map chip logic ids to npu-smi card ids from `npu-smi info -m`."""
    mapping: Dict[int, int] = {}
    for raw_line in (raw_output or "").splitlines():
        line = raw_line.strip()
        if not line or not re.match(r"^\d+\s+\d+\s+[-\d]+\s+", line):
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 4:
            continue
        npu_id, _chip_id, logic_id, _chip_name = parts[:4]
        if logic_id == "-":
            continue
        try:
            mapping[int(logic_id)] = int(npu_id)
        except ValueError:
            continue
    return mapping


def _resolve_npu_card_id(npu_smi_cmd: str, device_id: int) -> Optional[int]:
    """Resolve a logical device id to the card id expected by `npu-smi -i`."""
    mapping_output = _run_cmd(f"{npu_smi_cmd} info -m", timeout=15.0)
    mapping = _parse_npu_mapping(mapping_output)
    return mapping.get(device_id)


# ============================================================
# Internal helpers (reused from original implementation)
# ============================================================

def _find_cann_home() -> Optional[str]:
    """Find CANN Toolkit installation root."""
    env_home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if env_home and os.path.isdir(env_home):
        return env_home
    for p in _CANN_TOOLKIT_CANDIDATES:
        if os.path.isdir(p):
            return p
    return None


def _run_cmd(cmd: str, timeout: float = 10.0) -> str:
    """Run a shell command and return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _check_tool(name: str) -> Optional[str]:
    """Check if a command-line tool is available, return its path."""
    return shutil.which(name)


def _get_cann_version(cann_home: Optional[str]) -> str:
    """Get CANN version from ops/version.info."""
    if not cann_home:
        return "未知 (CANN 路径未找到)"
    version_file = os.path.join(cann_home, "ops", "version.info")
    if os.path.isfile(version_file):
        version = ""
        version_dir = ""
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("Version="):
                    version = line.split("=", 1)[1].strip()
                elif line.startswith("version_dir="):
                    version_dir = line.split("=", 1)[1].strip()
        if version and version_dir:
            return f"{version} ({version_dir})"
        return version or version_dir or "未知"
    return "未知 (version.info 未找到)"


def _normalize_api_parts(api_name: str) -> List[str]:
    name = (api_name or "").strip()
    for prefix in ("AscendC::", "ascendc::"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return [part for part in re.split(r"\s*::\s*", name) if part]


def _line_matches_api_symbol(api_name: str, line: str, *, context: str = "") -> bool:
    """Return True when a header line contains an exact symbol-like API match."""
    code = line.split("//", 1)[0].strip()
    if not code:
        return False

    parts = _normalize_api_parts(api_name)
    if not parts:
        return False
    short = parts[-1]
    escaped_short = re.escape(short)

    type_decl_re = re.compile(
        rf"\b(?:using|typedef|class|struct|enum(?:\s+class)?)\s+[^;={{}}]*\b{escaped_short}\b"
    )
    macro_re = re.compile(rf"^\s*#\s*define\s+{escaped_short}\b")
    token_follow_re = re.compile(
        rf"(?<![A-Za-z0-9_]){escaped_short}(?![A-Za-z0-9_])\s*(?:<|\()"
    )

    if type_decl_re.search(code) or macro_re.search(code):
        return True

    if len(parts) == 1:
        return bool(token_follow_re.search(code))

    qualified = r"\s*::\s*".join(re.escape(part) for part in parts)
    qualified_re = re.compile(
        rf"(?<![A-Za-z0-9_])(?:AscendC\s*::\s*)?{qualified}(?![A-Za-z0-9_])\s*(?:<|\()"
    )
    if qualified_re.search(code):
        return True

    owner = parts[-2]
    owner_context_re = re.compile(rf"\b(?:class|struct)\s+{re.escape(owner)}\b")
    return bool(token_follow_re.search(code) and owner_context_re.search(context))


def _search_api_in_headers(
    cann_home: Optional[str],
    api_name: str,
) -> List[str]:
    """Search headers for exact API-symbol matches. Returns 'file:line:content' strings."""
    if not cann_home:
        return []

    # Search directories for AscendC headers
    search_dirs = [
        os.path.join(cann_home, "aarch64-linux/ascendc/act/include"),
        os.path.join(cann_home, "aarch64-linux/include/ascendc"),
        os.path.join(cann_home, "include"),
    ]
    # Also search the opp include directory
    opp_path = os.environ.get("ASCEND_OPP_PATH", "").strip()
    if opp_path:
        search_dirs.append(os.path.join(opp_path, "include"))

    # Filter to existing directories
    search_dirs = [d for d in search_dirs if os.path.isdir(d)]

    if not search_dirs:
        return []

    results: List[str] = []

    for search_dir in search_dirs:
        for path in Path(search_dir).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            recent_context: List[str] = []
            for line_num, raw_line in enumerate(lines, start=1):
                context = "\n".join(recent_context[-8:])
                if _line_matches_api_symbol(api_name, raw_line, context=context):
                    results.append(f"{path}:{line_num}:{raw_line.strip()}")
                stripped = raw_line.strip()
                if stripped:
                    recent_context.append(stripped)

    return results


# ============================================================
# Tool 1: check_env
# ============================================================

def check_env() -> EnvCheckResult:
    """
    Check CANN environment configuration.

    Checks in Python (avoids ANSI color code parsing from bash scripts):
    - ASCEND_HOME_PATH and CANN version
    - ASCEND_OPP_PATH and vendors directory
    - CANN tools (msprof, cannsim, npu-smi)
    - Custom OPP packages
    - LD_LIBRARY_PATH configuration
    - Debug settings (ASCEND_SLOG_PRINT_TO_STDOUT)

    Returns:
        EnvCheckResult with structured fields
    """
    cann_home = _find_cann_home()
    cann_version = _get_cann_version(cann_home)
    opp_path = os.environ.get("ASCEND_OPP_PATH", "").strip()
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    errors = 0
    warnings = 0
    detail_lines: List[str] = []

    # 1. ASCEND_HOME_PATH
    if cann_home:
        detail_lines.append(f"[OK] ASCEND_HOME_PATH = {cann_home}")
    else:
        detail_lines.append("[ERROR] ASCEND_HOME_PATH 未设置")
        errors += 1

    # 2. CANN version
    detail_lines.append(f"[CANN 版本] {cann_version}")

    # 3. ASCEND_OPP_PATH
    if opp_path and os.path.isdir(opp_path):
        detail_lines.append(f"[OK] ASCEND_OPP_PATH = {opp_path}")
        vendors_dir = os.path.join(opp_path, "vendors")
        if os.path.isdir(vendors_dir):
            vendors = [d for d in os.listdir(vendors_dir)
                       if os.path.isdir(os.path.join(vendors_dir, d))]
            detail_lines.append(f"[OK] CANN Ops vendors: {len(vendors)} 个")
        else:
            detail_lines.append("[WARN] ASCEND_OPP_PATH/vendors 目录不存在")
            warnings += 1
    else:
        detail_lines.append("[WARN] ASCEND_OPP_PATH 未设置（编译时可跳过，运行时必需）")
        warnings += 1

    # 4. CANN tools
    tools = {}
    for tool in ["msprof", "cannsim", "npu-smi", "msopgen"]:
        path = _check_tool(tool)
        if path:
            tools[tool] = True
            detail_lines.append(f"[OK] {tool} 可用: {path}")
        else:
            tools[tool] = False
            if tool in ("npu-smi", "msopgen"):
                detail_lines.append(f"[WARN] {tool} 不可用")
                warnings += 1
            else:
                detail_lines.append(f"[INFO] {tool} 不可用（可选）")

    # 5. Custom OPP packages
    custom_opps: List[str] = []
    if cann_home:
        vendor_base = os.path.join(cann_home, "opp", "vendors")
        if os.path.isdir(vendor_base):
            for d in os.listdir(vendor_base):
                vendor_dir = os.path.join(vendor_base, d)
                if not os.path.isdir(vendor_dir):
                    continue
                # Check multiple possible lib paths
                op_api_lib = None
                for lib_path in ["op_api/lib", "op_impl/ai_core/tbe/op_api/lib"]:
                    candidate = os.path.join(vendor_dir, lib_path)
                    if os.path.isdir(candidate):
                        op_api_lib = candidate
                        break
                if op_api_lib:
                    custom_opps.append(d)
                    detail_lines.append(f"[OK] 自定义算子包 {d}: 已安装")
                    if op_api_lib not in ld_path:
                        detail_lines.append(
                            f"[WARN] LD_LIBRARY_PATH 未包含 {op_api_lib}"
                        )
                        warnings += 1
                else:
                    detail_lines.append(f"[INFO] vendors/{d}: 目录存在但无算子库")

    # 6. Debug settings
    slog = os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT", "")
    if slog == "1":
        detail_lines.append("[OK] 日志打屏已开启 (ASCEND_SLOG_PRINT_TO_STDOUT=1)")
    else:
        detail_lines.append("[INFO] 建议: export ASCEND_SLOG_PRINT_TO_STDOUT=1")

    all_passed = errors == 0
    return EnvCheckResult(
        all_passed=all_passed,
        cann_version=cann_version,
        cann_home=cann_home,
        opp_path=opp_path if opp_path else None,
        ld_library_path=ld_path,
        tools=tools,
        custom_opps=custom_opps,
        errors=errors,
        warnings=warnings,
        details="\n".join(detail_lines),
    )


# ============================================================
# Tool 2: query_npu_devices
# ============================================================

def query_npu_devices(
    device_id: Optional[int] = None,
    query_type: str = "info",
) -> NpuDeviceResult:
    """
    Query NPU device information via npu-smi.

    Args:
        device_id: Specific device ID (None = all devices)
        query_type: One of "info", "list", "memory", "temp", "power", "usages"

    Returns:
        NpuDeviceResult with raw output
    """
    query_type = query_type.lower().strip()
    npu_smi_path = _check_tool("npu-smi")
    if npu_smi_path:
        npu_smi_cmd = npu_smi_path
    else:
        npu_smi_cmd = "npu-smi"

    base_cmd = _NPU_QUERY_MAP.get(query_type, _NPU_QUERY_MAP["info"]).replace(
        "npu-smi", npu_smi_cmd, 1
    )
    cmd = base_cmd

    # `npu-smi info` and `npu-smi list` do not accept a bare `-i <card>`.
    # Only append scoped card ids for typed info queries after resolving the logical id.
    if device_id is not None and query_type in _NPU_INFO_SCOPED_TYPES:
        resolved_card_id = _resolve_npu_card_id(npu_smi_cmd, device_id)
        if resolved_card_id is not None:
            cmd = f"{base_cmd} -i {resolved_card_id}"

    output = _run_cmd(cmd, timeout=15.0)

    # If scoped lookup failed because the requested logical device could not be resolved,
    # fall back to the unscoped query instead of returning a deterministic CLI error.
    if not output and cmd != base_cmd:
        output = _run_cmd(base_cmd, timeout=15.0)

    available = bool(output) and "error" not in output.lower()

    return NpuDeviceResult(
        available=available,
        query_type=query_type,
        raw_output=output or "npu-smi 不可用或命令执行失败",
    )


# ============================================================
# Tool 3: check_api_exists
# ============================================================

def check_api_exists(api_name: str) -> ApiCheckResult:
    """
    Check if an Ascend C API exists in CANN header files.

    Searches in:
    - aarch64-linux/ascendc/act/include
    - aarch64-linux/include/ascendc
    - include/
    - ASCEND_OPP_PATH/include

    Args:
        api_name: API name (e.g., "DataCopy", "Muls", "AscendC::Cast")

    Returns:
        ApiCheckResult with match details
    """
    cann_home = _find_cann_home()
    results = _search_api_in_headers(cann_home, api_name)

    if not results:
        search_dirs = [
            os.path.join(cann_home, "aarch64-linux/ascendc/act/include") if cann_home else "N/A",
            os.path.join(cann_home, "aarch64-linux/include/ascendc") if cann_home else "N/A",
            os.path.join(cann_home, "include") if cann_home else "N/A",
        ]
        search_dirs = [d for d in search_dirs if d != "N/A"]
        opp_path = os.environ.get("ASCEND_OPP_PATH", "").strip()
        if opp_path:
            search_dirs.append(os.path.join(opp_path, "include"))

        return ApiCheckResult(
            found=False,
            api_name=api_name,
            header_files=[],
            matches=[],
            summary=(
                f"在 CANN 头文件中未找到 API: {api_name}\n"
                f"搜索路径: {', '.join(search_dirs)}\n"
                f"可能原因: API 名称不正确、或该 API 是宏/模板而非函数声明、或 CANN 版本不支持"
            ),
        )

    # Extract unique header file paths from results
    header_files = set()
    for r in results:
        parts = r.split(":", 2)
        if len(parts) >= 1:
            header_files.add(parts[0])

    return ApiCheckResult(
        found=True,
        api_name=api_name,
        header_files=sorted(header_files),
        matches=results[:5],
        summary=f"找到 {len(results)} 处匹配，涉及 {len(header_files)} 个头文件",
    )


# ============================================================
# Backward-compatible EnvCheckRetriever class
# ============================================================

class EnvCheckRetriever:
    """
    Wrapper for Ascend C environment checking.

    Provides three main capabilities as structured methods:
    1. check_env()       — CANN environment overview
    2. query_npu_devices() — NPU device query
    3. check_api_exists()  — API compatibility check

    Also retains the legacy retrieve() method for backward compatibility.
    """

    def __init__(self):
        self.cann_home = _find_cann_home()

    def is_available(self) -> bool:
        """Check if CANN environment exists."""
        return self.cann_home is not None

    def check_env(self) -> EnvCheckResult:
        """Check CANN environment configuration."""
        return check_env()

    def query_npu_devices(
        self,
        device_id: Optional[int] = None,
        query_type: str = "info",
    ) -> NpuDeviceResult:
        """Query NPU device information."""
        return query_npu_devices(device_id, query_type)

    def check_api_exists(self, api_name: str) -> ApiCheckResult:
        """Check if an API exists in CANN headers."""
        return check_api_exists(api_name)

    def retrieve(self, query: str) -> List[str]:
        """
        Legacy interface: perform environment check or API compatibility verification.

        Dispatches to the appropriate structured tool based on query keywords,
        then formats the result as a list of strings for backward compatibility.

        Args:
            query: Query string describing what to check:
                - "环境概览" / "check environment" -> full environment report
                - "NPU 设备" / "device status" -> NPU device info
                - "检查 API: XXX" / "is XXX available" -> API compatibility check
                - "诊断 XXX" -> troubleshooting hints

        Returns:
            List of formatted result strings
        """
        query_lower = query.lower().strip()

        # Troubleshooting hints
        hint = self._get_troubleshooting_hint(query)
        if hint:
            return [f"[诊断提示]\n{hint}"]

        # Environment overview
        if any(kw in query_lower for kw in [
            "环境", "environment", "check env", "检查环境", "概览", "overview"
        ]):
            result = check_env()
            return self._format_env_result(result)

        # NPU device info
        if any(kw in query_lower for kw in [
            "npu", "device", "设备", "npu-smi", "卡"
        ]):
            result = query_npu_devices()
            return [
                "=" * 50,
                "NPU 设备信息",
                "=" * 50,
                result.raw_output,
                "=" * 50,
            ]

        # API compatibility check
        api_name = ""
        match = re.search(r"(?:检查\s*api[:：]?\s*|check\s*(?:if\s*)?(?:api\s*)?[:：]?\s*|is\s+)(\w+)", query_lower)
        if match:
            api_name = match.group(1)
        if not api_name:
            match = re.search(r"(\w+)\s*(?:api|函数|function|算子)", query_lower)
            if match:
                api_name = match.group(1)
        if not api_name and re.match(r"^[A-Za-z_]\w{2,}$", query.strip()):
            api_name = query.strip()

        if api_name:
            result = check_api_exists(api_name)
            return self._format_api_result(result)

        # Fallback: treat query as API name
        if query.strip():
            result = check_api_exists(query.strip())
            return self._format_api_result(result)

        # Default to environment overview
        result = check_env()
        return self._format_env_result(result)

    @staticmethod
    def _format_env_result(result: EnvCheckResult) -> List[str]:
        """Format EnvCheckResult for display (backward compatibility)."""
        lines = [
            "=" * 50,
            "Ascend C 环境检查报告",
            "=" * 50,
            "",
            f"[CANN 版本] {result.cann_version}",
            f"[CANN 路径] {result.cann_home or '未找到'}",
            "",
        ]
        lines.append("[环境变量]")
        lines.append(f"  ASCEND_HOME_PATH = {result.cann_home or '(未设置)'}")
        lines.append(f"  ASCEND_OPP_PATH = {result.opp_path or '(未设置)'}")
        lines.append(f"  LD_LIBRARY_PATH = {result.ld_library_path[:80]}...")
        lines.append("")
        lines.append("[CANN 工具]")
        for tool, ok in result.tools.items():
            status = "可用" if ok else "不可用"
            lines.append(f"  {tool}: {status}")
        lines.append("")
        if result.custom_opps:
            lines.append(f"[自定义算子包] {', '.join(result.custom_opps)}")
        else:
            lines.append("[自定义算子包] 未安装")
        lines.append("")
        lines.append(f"[结果] {'通过' if result.all_passed else f'发现 {result.errors} 个错误'}")
        if result.warnings > 0:
            lines.append(f"  警告: {result.warnings} 个")
        lines.append("")
        lines.append("=" * 50)
        return lines

    @staticmethod
    def _format_api_result(result: ApiCheckResult) -> List[str]:
        """Format ApiCheckResult for display (backward compatibility)."""
        lines = [
            "=" * 50,
            f"API 兼容性检查: {result.api_name}",
            "=" * 50,
        ]
        if result.found:
            lines.append(f"[结果] 找到 {len(result.matches)} 处匹配")
            lines.append(f"[头文件] {', '.join(result.header_files)}")
            lines.append("")
            for m in result.matches:
                lines.append(f"  {m}")
        else:
            lines.append(f"[结果] API '{result.api_name}' 在头文件中未找到")
            lines.append("")
            lines.append(result.summary)
        lines.append("")
        lines.append("=" * 50)
        return lines

    @staticmethod
    def _get_troubleshooting_hint(query: str) -> Optional[str]:
        """Provide troubleshooting hints based on query keywords."""
        query_lower = query.lower()
        hints = {
            "561003": "错误 561003: Kernel 查找失败。检查 LD_LIBRARY_PATH 是否包含 op_api/lib，确认算子包已正确安装。",
            "561107": "错误 561107: 环境变量配置错误。请使用 source set_env.sh 配置，不要手动 export ASCEND_OPP_PATH。",
            "device not found": "NPU 设备未找到。检查: 1) npu-smi list 是否识别设备 2) 驱动是否安装 3) 容器是否正确映射设备",
            "libascend": "运行时库依赖问题。export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/runtime/lib64:$LD_LIBRARY_PATH",
        }
        for keyword, hint in hints.items():
            if keyword in query_lower:
                return hint
        return None


# ============================================================
# Convenience functions
# ============================================================

def check_env_convenience(query: str = "") -> List[str]:
    """
    Convenience function for environment check (legacy interface).

    Args:
        query: What to check (empty for full overview)

    Returns:
        List of formatted result strings
    """
    retriever = EnvCheckRetriever()
    return retriever.retrieve(query or "check environment")


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "check environment"
    retriever = EnvCheckRetriever()
    if not retriever.is_available():
        print("[WARN] CANN 环境未找到")
    for t in retriever.retrieve(q):
        print(t)
