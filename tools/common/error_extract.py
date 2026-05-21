"""Layered build/eval error extraction: root_cause (compile/API) vs symptom (pipeline tail)."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

ROOT_CAUSE_MARKERS: Tuple[str, ...] = (
    "error: no member named",
    "cannot convert",
    "undefined reference",
    "has no member named",
    "Kernel Compilation Error",
    "build ops lib error",
    "fatal error:",
    "undeclared identifier",
)

COMPILE_LINE_RE = re.compile(
    r"\.(?:cpp|cc|c|h|hpp):\d+:\d+:\s*error:",
    re.IGNORECASE,
)

SYMPTOM_MARKERS: Tuple[str, ...] = (
    "CMake Error",
    "CPack Error",
    "file INSTALL cannot find",
    "binary/config",
    "op_host/CMakeLists.txt",
    "Configuring incomplete",
    "txt bundle missing blocks",
    "ValueError:",
    "No rule to make target",
    "gmake: ***",
    "# returncode:",
)

CORE_ERROR_MARKERS: Tuple[str, ...] = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "CalledProcessError:",
    "CMake Error",
    "error:",
    "ERROR",
    "ERR",
    "No such file or directory",
    "not found",
    "failed",
)


def read_log_tail_text(text: str, max_lines: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _non_empty_lines(text: str) -> List[str]:
    return [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]


def _line_is_root_cause(ln: str) -> bool:
    low = ln.lower()
    if any(m.lower() in low for m in ROOT_CAUSE_MARKERS):
        return True
    if COMPILE_LINE_RE.search(ln):
        return True
    if ": error:" in low and "cmake error" not in low:
        return True
    return False


def _line_is_symptom(ln: str) -> bool:
    return any(m in ln for m in SYMPTOM_MARKERS)


def _window(lines: List[str], center: int, before: int = 3, after: int = 3) -> str:
    start = max(0, center - before)
    end = min(len(lines), center + after + 1)
    return "\n".join(lines[start:end])


def extract_root_cause(log_text: str) -> str:
    """
    Prefer earliest compile/API-class error in the log excerpt, not the final CPack line.
    """
    lines = _non_empty_lines(log_text)
    if not lines:
        return ""

    root_idxs = [i for i, ln in enumerate(lines) if _line_is_root_cause(ln)]
    if root_idxs:
        return _window(lines, root_idxs[0], before=2, after=3)

    return ""


def extract_symptom(log_text: str, *, fallback_text: str = "") -> str:
    """
    Pipeline-level / last-stage failure (CPack, INSTALL, opbuild, ValueError, etc.).
    """
    lines = _non_empty_lines(log_text)
    if not lines and fallback_text:
        lines = _non_empty_lines(fallback_text)
    if not lines:
        return ""

    symptom_idxs = [i for i, ln in enumerate(lines) if _line_is_symptom(ln)]
    if symptom_idxs:
        return _window(lines, symptom_idxs[-1], before=4, after=5)

    return extract_core_error(log_text or fallback_text)


def extract_core_error(text: str) -> str:
    """Legacy short excerpt around the last error marker (symptom-oriented)."""
    if not text:
        return ""
    lines = _non_empty_lines(text)
    if not lines:
        return ""

    idxs = [i for i, ln in enumerate(lines) if any(m in ln for m in CORE_ERROR_MARKERS)]
    if not idxs:
        return "\n".join(lines[-20:])
    start = max(0, idxs[-1] - 6)
    end = min(len(lines), idxs[-1] + 8)
    return "\n".join(lines[start:end])


def anchor_from_excerpt(text: str, max_len: int = 256) -> str:
    """Stable short anchor for manifest / comparison."""
    raw = (text or "").strip().replace("\r\n", "\n")
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    pick: List[str] = []
    for ln in lines:
        if any(
            x in ln
            for x in (
                "error:",
                "Error",
                "CMake",
                "CPack",
                "undefined reference",
                "fatal:",
                "ValueError",
                "Kernel Compilation",
            )
        ):
            pick.append(ln[:200])
        if sum(len(x) for x in pick) >= max_len:
            break
    blob = " | ".join(pick) if pick else (lines[-1][:200] if lines else raw[:200])
    return blob[:max_len]


def format_layered_correctness_info(
    *,
    root_cause: str,
    symptom: str,
    exc_fallback: str = "",
) -> str:
    parts: List[str] = []
    root = (root_cause or "").strip()
    sym = (symptom or "").strip()
    if not root and not sym and exc_fallback:
        sym = exc_fallback.strip()
    if not root and sym and not sym.startswith("==="):
        root = ""
    if root:
        parts.append("=== root_cause ===")
        parts.append(root)
        parts.append("")
    if sym:
        parts.append("=== symptom ===")
        parts.append(sym)
    if not parts and exc_fallback:
        return exc_fallback.strip()
    return "\n".join(parts).strip() + "\n"


def parse_layered_correctness_info(text: str) -> Tuple[str, str]:
    """Return (root_cause, symptom) from formatted correctness_info."""
    raw = text or ""
    if "=== root_cause ===" not in raw and "=== symptom ===" not in raw:
        sym = extract_symptom("", fallback_text=raw)
        root = extract_root_cause(raw)
        if root and root != sym:
            return root, sym or raw.strip()
        return "", raw.strip()

    root = ""
    symptom = ""
    if "=== root_cause ===" in raw:
        after_root = raw.split("=== root_cause ===", 1)[1]
        if "=== symptom ===" in after_root:
            root, rest = after_root.split("=== symptom ===", 1)
            symptom = rest
        else:
            root = after_root
    elif "=== symptom ===" in raw:
        symptom = raw.split("=== symptom ===", 1)[1]

    return root.strip(), symptom.strip()


def build_layered_errors_from_log_text(
    log_text: str,
    *,
    exc_fallback: str = "",
) -> Tuple[str, str, str]:
    """
    Returns (root_cause, symptom, formatted_correctness_info).
    """
    root = extract_root_cause(log_text)
    symptom = extract_symptom(log_text, fallback_text=exc_fallback or log_text)
    if not symptom and exc_fallback:
        symptom = extract_core_error(exc_fallback)
    if not root and not symptom:
        symptom = extract_core_error(log_text) or (exc_fallback or "").strip()
    formatted = format_layered_correctness_info(
        root_cause=root,
        symptom=symptom,
        exc_fallback=exc_fallback,
    )
    return root, symptom, formatted
