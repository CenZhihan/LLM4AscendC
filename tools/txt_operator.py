#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys
import warnings
from dataclasses import dataclass

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class TxtBundle:
    project_json_src: str
    host_tiling_src: str
    host_operator_src: str
    kernel_src: str
    python_bind_src: str
    model_src: str | None = None
    eval_src: str | None = None


_VAR_NAMES = [
    "project_json_src",
    "host_tiling_src",
    "host_operator_src",
    "kernel_src",
    "python_bind_src",
    "model_src",
    "eval_src",
]


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.splitlines()[1:]
        s = "\n".join(s)
    if s.rstrip().endswith("```"):
        s = "\n".join(s.splitlines()[:-1])
    return s


def _extract_triple_quoted_var(src: str, var: str) -> str | None:
    # Matches: var = ''' ... '''  or var = """ ... """
    # non-greedy, dotall, allow spaces.
    pat = re.compile(
        rf"{re.escape(var)}\s*=\s*(?P<q>'''|\"\"\")(?P<body>[\s\S]*?)(?P=q)",
        re.MULTILINE,
    )
    m = pat.search(src)
    if not m:
        return None
    return m.group("body")


def parse_txt_bundle(txt_path: pathlib.Path) -> TxtBundle:
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    src = _strip_code_fence(raw)

    vals: dict[str, str | None] = {k: _extract_triple_quoted_var(src, k) for k in _VAR_NAMES}

    required = ["project_json_src", "host_tiling_src", "host_operator_src", "kernel_src", "python_bind_src"]
    missing = [k for k in required if not vals.get(k)]
    if missing:
        raise ValueError(f"txt bundle missing blocks: {missing}")

    return TxtBundle(
        project_json_src=str(vals["project_json_src"]),
        host_tiling_src=str(vals["host_tiling_src"]),
        host_operator_src=str(vals["host_operator_src"]),
        kernel_src=str(vals["kernel_src"]),
        python_bind_src=str(vals["python_bind_src"]),
        model_src=vals.get("model_src") or None,
        eval_src=vals.get("eval_src") or None,
    )


def infer_op_name(project_json_src: str) -> str:
    obj = json.loads(project_json_src)
    if not isinstance(obj, list) or not obj or "op" not in obj[0]:
        raise ValueError("project_json_src must be a JSON list with first element containing 'op'")
    return str(obj[0]["op"])


def infer_op_snake(kernel_src: str) -> str:
    # Prefer the exported kernel function name.
    m = re.search(r"__global__\s+__aicore__\s+void\s+([a-zA-Z0-9_]+)\s*\(", kernel_src)
    if m:
        return m.group(1)
    # Fallback: any extern "C" void name
    m2 = re.search(r'extern\s+"C"\s+.*void\s+([a-zA-Z0-9_]+)\s*\(', kernel_src)
    if m2:
        return m2.group(1)
    raise ValueError("cannot infer op_snake from kernel_src")


def camel_to_key(name: str) -> str:
    # simple default: CamelCase -> snake-ish key
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _warn_if_json_op_mismatch(*, op_key: str, project_json_src: str) -> None:
    """Optional: project_json op name (strip Custom) should match txt filename stem as MKB key."""
    try:
        op_name = infer_op_name(project_json_src)
    except Exception:
        return
    json_stem = op_name[:-6] if op_name.endswith("Custom") else op_name
    inferred = camel_to_key(json_stem)
    if inferred != op_key:
        warnings.warn(
            f"project_json op {op_name!r} implies MKB key {inferred!r}, but txt filename stem is {op_key!r}; "
            "using filename stem for MKB reference.",
            UserWarning,
            stacklevel=2,
        )


def materialize_operator_from_txt(
    *,
    out_dir: pathlib.Path,
    txt_path: pathlib.Path,
    soc: str,
    module_name: str | None = None,
) -> pathlib.Path:
    """
    Create an operator directory that matches our evaluation contract:
      operator.json, op_host/, op_kernel/, pybind/op.cpp, eval/spec.py

    MKB op_key is taken from the txt filename stem (e.g. output/layer_norm.txt -> layer_norm).
    Golden reference is always vendored: vendor/mkb/reference/{category}/{op_key}.py
    """
    from vendor.mkb.ref_paths import get_ref_py_path

    bundle = parse_txt_bundle(txt_path)
    op_key = txt_path.stem
    try:
        get_ref_py_path(op_key)
    except KeyError as e:
        raise ValueError(str(e)) from e
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e

    _warn_if_json_op_mismatch(op_key=op_key, project_json_src=bundle.project_json_src)

    op_name = infer_op_name(bundle.project_json_src)
    op_snake = infer_op_snake(bundle.kernel_src)

    mod = module_name or f"llm4ascendc_ops_{op_key}"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "op_host").mkdir(parents=True, exist_ok=True)
    (out_dir / "op_kernel").mkdir(parents=True, exist_ok=True)
    (out_dir / "pybind").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)

    # operator.json
    operator_obj = {
        "op_key": op_key,
        "op_name": op_name,
        "op_snake": op_snake,
        "soc": soc,
        "project_json": json.loads(bundle.project_json_src),
        "pybind": {"module_name": mod},
    }
    (out_dir / "operator.json").write_text(json.dumps(operator_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # host/kernel sources
    (out_dir / "op_host" / f"{op_snake}_tiling.h").write_text(bundle.host_tiling_src, encoding="utf-8")
    (out_dir / "op_host" / f"{op_snake}.cpp").write_text(bundle.host_operator_src, encoding="utf-8")
    (out_dir / "op_kernel" / f"{op_snake}.cpp").write_text(bundle.kernel_src, encoding="utf-8")

    # pybind
    (out_dir / "pybind" / "op.cpp").write_text(bundle.python_bind_src, encoding="utf-8")

    # eval
    if bundle.eval_src:
        (out_dir / "eval" / "spec.py").write_text(bundle.eval_src, encoding="utf-8")
    elif bundle.model_src:
        # Vendored MKB reference + execute_template (same behavior as MKB correctness.py).
        model_literal = bundle.model_src
        eval_py = f"""#!/usr/bin/env python3
import os
import sys
import pathlib
import torch
import torch_npu  # noqa: F401


def _find_repo_root() -> pathlib.Path:
    p = pathlib.Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vendor" / "mkb" / "correctness.py").is_file():
            return parent
    raise RuntimeError(
        "Cannot find LLM4AscendC repo root (expected vendor/mkb/correctness.py on a parent path)."
    )


def main() -> int:
    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vendor.mkb.correctness import execute_template
    from vendor.mkb.ref_paths import get_ref_py_path

    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "{mod}")
    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    op_key = {op_key!r}
    ref_path = get_ref_py_path(op_key)
    ref_src = ref_path.read_text(encoding="utf-8")

    ctx = {{"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}}
    exec(ref_src, ctx, ctx)
    exec({model_literal!r}, ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")
    for k in ("Model", "get_inputs", "get_init_inputs"):
        if k not in ctx:
            raise RuntimeError("MKB reference for {repr(op_key)} must define " + k)

    device = torch.device("npu:0")
    synchronize = torch_npu.npu.synchronize

    correctness, info = execute_template(synchronize, device, ctx)
    if not correctness:
        print(info or "[FAIL] Output mismatch")
        return 1
    print("[eval] allclose=True [ref=MKB " + op_key + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
        (out_dir / "eval" / "spec.py").write_text(eval_py, encoding="utf-8")
    else:
        # Smoke test only: import module; no correctness check.
        smoke = f"""#!/usr/bin/env python3
import os
import torch_npu  # noqa: F401

def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "{mod}")
    __import__(module_name)
    print(f"[smoke] imported module={{module_name}}")
    print("[smoke] no model_src; skip correctness check")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
        (out_dir / "eval" / "spec.py").write_text(smoke, encoding="utf-8")

    # copy original txt for traceability
    (out_dir / "source_bundle.txt").write_text(txt_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return out_dir

