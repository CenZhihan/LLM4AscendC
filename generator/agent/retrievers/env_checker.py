"""
Environment Checker for Ascend C kernel development agent.

Adapts the CANN skill's environment checking capability (scripts/check_env.sh,
scripts/npu_info.sh) into a Python retriever for agent use.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


# Default CANN paths
_CANN_TOOLKIT_CANDIDATES = [
    "/usr/local/Ascend/ascend-toolkit/latest",
    os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
]


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
    path = shutil.which(name)
    return path


class EnvCheckRetriever:
    """
    Wrapper for Ascend C environment checking.

    Provides two main capabilities:
    1. Environment overview: CANN version, env vars, tools, NPU devices
    2. API compatibility check: verify if a specific API exists in CANN headers
    """

    def __init__(self):
        self.cann_home = _find_cann_home()

    def is_available(self) -> bool:
        """Check if CANN environment exists."""
        return self.cann_home is not None

    def _get_cann_version(self) -> str:
        """Get CANN version from ops/version.info."""
        if not self.cann_home:
            return "未知 (CANN 路径未找到)"
        version_file = os.path.join(self.cann_home, "ops", "version.info")
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

    def _check_env_vars(self) -> List[str]:
        """Check key environment variables."""
        lines: List[str] = []
        for var in ["ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH"]:
            val = os.environ.get(var, "")
            if val:
                lines.append(f"[OK] {var} = {val}")
            else:
                lines.append(f"[WARN] {var} 未设置")
        return lines

    def _check_cann_tools(self) -> List[str]:
        """Check availability of key CANN tools."""
        lines: List[str] = []
        tools = {
            "msopgen": "算子代码生成工具",
            "msprof": "性能分析工具",
            "npu-smi": "NPU 设备管理工具",
        }
        for tool, desc in tools.items():
            path = _check_tool(tool)
            if path:
                lines.append(f"[OK] {tool} ({desc}): {path}")
            else:
                lines.append(f"[WARN] {tool} ({desc}) 不可用")
        return lines

    def _get_npu_info(self) -> str:
        """Get NPU device information."""
        output = _run_cmd("npu-smi info 2>/dev/null")
        if output:
            return output
        # Try with full path
        npu_smi = _check_tool("npu-smi")
        if npu_smi:
            return _run_cmd(f"{npu_smi} info 2>/dev/null")
        return "npu-smi 不可用，无法获取 NPU 设备信息"

    def _check_custom_opps(self) -> List[str]:
        """Check custom OPP package status."""
        lines: List[str] = []
        opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").strip()
        if not opp_path:
            opp_path = os.environ.get("ASCEND_OPP_PATH", "").strip()

        if opp_path and os.path.isdir(opp_path):
            vendor_dir = os.path.join(opp_path, "vendors")
            if os.path.isdir(vendor_dir):
                vendors = [d for d in os.listdir(vendor_dir)
                           if os.path.isdir(os.path.join(vendor_dir, d))]
                lines.append(f"[OK] OPP vendors: {', '.join(vendors) if vendors else '无'}")
            else:
                lines.append("[WARN] vendors 目录不存在")

            # Check lib paths
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if "op_api/lib" in ld_path:
                lines.append("[OK] LD_LIBRARY_PATH 包含 op_api/lib")
            else:
                lines.append("[WARN] LD_LIBRARY_PATH 可能未包含 op_api/lib")
        else:
            lines.append("[WARN] 自定义算子包路径未设置")
        return lines

    def _search_api_in_headers(self, api_name: str) -> List[str]:
        """Search for API in CANN header files."""
        if not self.cann_home:
            return ["[ERROR] CANN 路径未找到，无法搜索 API"]

        # Search directories for AscendC headers
        search_dirs = [
            os.path.join(self.cann_home, "aarch64-linux/ascendc/act/include"),
            os.path.join(self.cann_home, "aarch64-linux/include/ascendc"),
            os.path.join(self.cann_home, "include"),
        ]
        # Also search the opp include directory
        opp_path = os.environ.get("ASCEND_OPP_PATH", "").strip()
        if opp_path:
            search_dirs.append(os.path.join(opp_path, "include"))

        # Filter to existing directories
        search_dirs = [d for d in search_dirs if os.path.isdir(d)]

        if not search_dirs:
            return ["[WARN] 未找到 AscendC 头文件目录"]

        results: List[str] = []
        api_name_stripped = api_name.strip().lstrip("AscendC::").lstrip("ascendc::")

        # Search for the API name in headers
        for search_dir in search_dirs:
            try:
                grep_cmd = f"grep -rn --include='*.h' --include='*.hpp' -i '{api_name_stripped}' '{search_dir}' 2>/dev/null || true"
                output = _run_cmd(grep_cmd, timeout=15.0)
                if output:
                    for line in output.splitlines():
                        results.append(line)
            except Exception:
                pass

        if not results:
            return [
                f"[未找到] 在 CANN 头文件中未找到 API: {api_name}",
                f"搜索路径: {', '.join(search_dirs)}",
                "可能原因: API 名称不正确、或该 API 是宏/模板而非函数声明、或 CANN 版本不支持",
            ]

        return results

    def _get_troubleshooting_hint(self, query: str) -> Optional[str]:
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

    def _build_env_overview(self) -> List[str]:
        """Build comprehensive environment overview report."""
        lines: List[str] = []
        lines.append("=" * 50)
        lines.append("Ascend C 环境检查报告")
        lines.append("=" * 50)
        lines.append("")

        # CANN version
        lines.append(f"[CANN 版本] {self._get_cann_version()}")
        lines.append(f"[CANN 路径] {self.cann_home or '未找到'}")
        lines.append("")

        # Environment variables
        lines.append("[环境变量]")
        lines.extend(self._check_env_vars())
        lines.append("")

        # CANN tools
        lines.append("[CANN 工具]")
        lines.extend(self._check_cann_tools())
        lines.append("")

        # NPU info
        lines.append("[NPU 设备]")
        npu_info = self._get_npu_info()
        for npu_line in npu_info.splitlines():
            lines.append(f"  {npu_line}")
        lines.append("")

        # Custom OPPs
        lines.append("[自定义算子包]")
        lines.extend(self._check_custom_opps())
        lines.append("")

        lines.append("=" * 50)
        return lines

    def _build_api_check_report(self, query: str, api_name: str) -> List[str]:
        """Build API compatibility check report."""
        lines: List[str] = []
        lines.append("=" * 50)
        lines.append(f"API 兼容性检查: {api_name}")
        lines.append("=" * 50)
        lines.append(f"[CANN 版本] {self._get_cann_version()}")
        lines.append("")

        results = self._search_api_in_headers(api_name)
        if results:
            if results[0].startswith("[未找到]"):
                lines.append(f"[结果] API '{api_name}' 在头文件中未找到")
                lines.append("")
                for r in results:
                    lines.append(f"  {r}")
            else:
                # Found matches - summarize
                header_files = set()
                for r in results:
                    parts = r.split(":", 2)
                    if len(parts) >= 2:
                        header_files.add(parts[0])

                lines.append(f"[结果] 找到 {len(results)} 处匹配")
                lines.append(f"[头文件] {', '.join(sorted(header_files))}")
                lines.append("")
                # Show first few matches
                for r in results[:5]:
                    lines.append(f"  {r}")
                if len(results) > 5:
                    lines.append(f"  ... 还有 {len(results) - 5} 处匹配")
        lines.append("")
        lines.append("=" * 50)
        return lines

    def retrieve(self, query: str) -> List[str]:
        """
        Perform environment check or API compatibility verification.

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

        # Check for troubleshooting keywords first
        hint = self._get_troubleshooting_hint(query)
        if hint:
            return [f"[诊断提示]\n{hint}"]

        # Environment overview
        if any(kw in query_lower for kw in [
            "环境", "environment", "check env", "检查环境", "概览", "overview"
        ]):
            return self._build_env_overview()

        # NPU device info
        if any(kw in query_lower for kw in [
            "npu", "device", "设备", "npu-smi", "卡"
        ]):
            npu_info = self._get_npu_info()
            return [
                "=" * 50,
                "NPU 设备信息",
                "=" * 50,
                npu_info,
                "=" * 50,
            ]

        # API compatibility check
        api_name = ""
        # Pattern: "检查 API: XXX" / "check if XXX exists" / "is XXX available"
        match = re.search(r"(?:检查\s*api[:：]?\s*|check\s*(?:if\s*)?(?:api\s*)?[:：]?\s*|is\s+)(\w+)", query_lower)
        if match:
            api_name = match.group(1)
        # Also try: "API XXX" / "XXX API"
        if not api_name:
            match = re.search(r"(\w+)\s*(?:api|函数|function|算子)", query_lower)
            if match:
                api_name = match.group(1)
        # If query is short and looks like an API name (camelCase, PascalCase)
        if not api_name and re.match(r"^[A-Za-z_]\w{2,}$", query.strip()):
            api_name = query.strip()

        if api_name:
            return self._build_api_check_report(query, api_name)

        # Fallback: treat query as API name if it's not empty
        if query.strip():
            return self._build_api_check_report(query, query.strip())

        # Default to environment overview
        return self._build_env_overview()


def check_env(query: str = "") -> List[str]:
    """
    Convenience function for environment check.

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
