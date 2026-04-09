#!/usr/bin/env python3
"""
Compare kernelbench165 txt python_bind_src entry-point signature with vendor MKB reference
(Model.forward + get_inputs). Reports arity/type contract mismatches that cause pybind errors.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT_DIR = ROOT / "output" / "kernelbench165_txt"
REF_ROOT = ROOT / "vendor" / "mkb" / "reference"


def extract_python_bind_src(txt: str) -> str | None:
    m = re.search(
        r'python_bind_src\s*=\s*"""(.*?)"""',
        txt,
        re.DOTALL,
    )
    if not m:
        return None
    return m.group(1)


def find_pybind_entry_func(bind: str) -> str | None:
    m = re.search(r'm\.def\s*\(\s*"[^"]+"\s*,\s*&(\w+)\s*[,)]', bind)
    return m.group(1) if m else None


def split_cpp_params(inner: str) -> list[str]:
    inner = re.sub(r"/\*[\s\S]*?\*/", "", inner)
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in inner:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def classify_cpp_param(p: str) -> str:
    p = re.sub(r"//.*", "", p).strip()
    if not p:
        return "unknown"
    if "optional" in p.lower():
        return "optional"
    if re.search(r"\bat::Tensor\b|const\s+at::Tensor", p):
        return "tensor"
    if re.search(r"\bTensor\b", p) and "at::" not in p:
        # rare
        return "tensor"
    if re.search(r"\b(double|float)\b", p):
        return "float"
    if re.search(r"\b(int64_t|int32_t|int)\b", p):
        return "int"
    if re.search(r"\bbool\b", p):
        return "bool"
    return "other"


def _cpp_param_list_after_name(bind: str, func_name: str) -> str | None:
    """
    Return text inside the first (...) that follows func_name (balanced parens, skips //, /* */, strings).
    Do not use regex .*?\\) — the first ') {' in the *body* (e.g. if (...) {) would truncate the param list.
    """
    pos = bind.find(func_name)
    if pos < 0:
        return None
    i = bind.find("(", pos + len(func_name))
    if i < 0:
        return None
    depth = 0
    start_open = i
    i = start_open
    n = len(bind)
    while i < n:
        c = bind[i]
        if c == "/" and i + 1 < n:
            nxt = bind[i + 1]
            if nxt == "/":
                nl = bind.find("\n", i)
                i = nl + 1 if nl >= 0 else n
                continue
            if nxt == "*":
                end = bind.find("*/", i + 2)
                if end < 0:
                    return None
                i = end + 2
                continue
        if c in "\"'":
            quote = c
            i += 1
            while i < n:
                if bind[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if bind[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return bind[start_open + 1 : i]
        i += 1
    return None


def extract_impl_signature(bind: str, func_name: str) -> tuple[list[str], str] | None:
    inner = _cpp_param_list_after_name(bind, func_name)
    if inner is None:
        return None
    parts = split_cpp_params(inner)
    kinds = [classify_cpp_param(x) for x in parts]
    return kinds, inner


def count_forward_positional_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int | None, bool]:
    args = node.args
    npos = len(args.args) - 1  # minus self
    if args.vararg is not None or args.kwonlyargs:
        return None, True
    if args.kw_defaults or args.defaults:
        # defaults ok if we only compare min - skip
        pass
    return npos, False


def get_inputs_list_len(tree: ast.Module) -> tuple[int | None, bool]:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "get_inputs":
            for st in n.body:
                if isinstance(st, ast.Return) and st.value is not None:
                    elts = None
                    if isinstance(st.value, (ast.List, ast.Tuple)):
                        elts = st.value.elts
                    if elts is not None:
                        return len(elts), False
            return None, True
    return None, True


def parse_reference(ref_path: pathlib.Path) -> tuple[int | None, int | None, str]:
    src = ref_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fwd_n = None
    gi_n = None
    notes = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "Model":
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name == "forward":
                    c, vague = count_forward_positional_args(m)
                    fwd_n = c
                    if vague:
                        notes.append("forward_has_varargs")
    gi_n, vague_gi = get_inputs_list_len(tree)
    if vague_gi:
        notes.append("get_inputs_dynamic")
    return fwd_n, gi_n, ";".join(notes)


def main() -> int:
    rows: list[tuple[str, str]] = []
    mismatches: list[dict] = []

    for txt_path in sorted(TXT_DIR.glob("*.txt")):
        op_key = txt_path.stem
        try:
            from vendor.mkb.ref_paths import get_ref_py_path
        except Exception:
            sys.path.insert(0, str(ROOT))
            from vendor.mkb.ref_paths import get_ref_py_path

        try:
            ref_path = get_ref_py_path(op_key)
        except Exception as e:
            mismatches.append(
                {
                    "op_key": op_key,
                    "issue": f"no_reference: {e}",
                }
            )
            continue

        txt = txt_path.read_text(encoding="utf-8", errors="replace")
        bind = extract_python_bind_src(txt)
        if not bind:
            mismatches.append({"op_key": op_key, "issue": "no_python_bind_src"})
            continue

        fn = find_pybind_entry_func(bind)
        if not fn:
            mismatches.append({"op_key": op_key, "issue": "no_m_def"})
            continue

        sig = extract_impl_signature(bind, fn)
        if not sig:
            mismatches.append({"op_key": op_key, "issue": f"no_impl_body:{fn}"})
            continue

        kinds, _inner = sig
        cpp_n = len(kinds)
        n_tensor = sum(1 for k in kinds if k == "tensor")
        n_float = sum(1 for k in kinds if k == "float")
        n_int = sum(1 for k in kinds if k == "int")

        fwd_n, gi_n, note = parse_reference(ref_path)

        row = {
            "op_key": op_key,
            "cpp_params": cpp_n,
            "cpp_kinds": ",".join(kinds),
            "fwd_args": fwd_n,
            "get_inputs_n": gi_n,
            "note": note,
        }

        ok = True
        if fwd_n is not None and gi_n is not None and fwd_n != gi_n:
            row["issue"] = "forward_vs_get_inputs_count"
            ok = False
        if gi_n is not None and gi_n != cpp_n:
            row["issue"] = "get_inputs_vs_cpp_count"
            ok = False
        if fwd_n is not None and fwd_n != cpp_n:
            row["issue"] = "forward_vs_cpp_count"
            ok = False

        if not ok:
            mismatches.append(row)

    out_path = ROOT / "artifacts" / "pybind_reference_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(
        json.dumps(mismatches, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[audit] mismatches: {len(mismatches)} -> {out_path}")
    for m in mismatches[:60]:
        print(m)
    if len(mismatches) > 60:
        print(f"... and {len(mismatches) - 60} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
