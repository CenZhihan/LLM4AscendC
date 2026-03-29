#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass


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
    """
    bundle = parse_txt_bundle(txt_path)
    op_name = infer_op_name(bundle.project_json_src)
    op_snake = infer_op_snake(bundle.kernel_src)
    op_key = camel_to_key(op_name)

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
    else:
        if bundle.model_src:
            mkb_root = (pathlib.Path(__file__).resolve().parents[1]).resolve()

            if op_key == "matmul_with_transposed_b_custom":
                # C = A @ B^T with A:[M,K], B:[N,K] fp16 in; reference selectable via env:
                #   LLM4ASCENDC_MKB_REFERENCE=0 (default) — small tensors, torch.matmul(A,B.T) on CPU ref side as fp32
                #   LLM4ASCENDC_MKB_REFERENCE=1 — vendored MKB reference/matmul/matmul_with_transposed_b.py + execute_template
                eval_py = f"""#!/usr/bin/env python3
import os
import sys
import pathlib
import torch
import torch_npu  # noqa: F401

def _sync():
    try:
        torch_npu.npu.synchronize()
    except Exception:
        pass

def set_seed(seed: int):
    torch.manual_seed(seed)

def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "{mod}")
    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    ctx = {{"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}}
    _ROOT = pathlib.Path(__file__).resolve().parents[4]
    mkb_ref_path = _ROOT / "mkb_reference" / "matmul" / "matmul_with_transposed_b.py"
    mkb_repo = _ROOT.parent / "LLM" / "MultiKernelBench-main"

    use_mkb = os.environ.get("LLM4ASCENDC_MKB_REFERENCE", "0").strip() == "1"
    execute_template = None
    if use_mkb:
        if mkb_repo.exists():
            sys.path.insert(0, str(mkb_repo))
        from utils.correctness import execute_template  # type: ignore

        ref_src = mkb_ref_path.read_text(encoding="utf-8")
        exec(ref_src, ctx, ctx)
    else:
        M, K, N = 256, 512, 384

        def get_init_inputs():
            return []

        def get_inputs():
            torch.manual_seed(0)
            A = torch.randn(M, K, dtype=torch.float16)
            B = torch.randn(N, K, dtype=torch.float16)
            return [A, B]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                return torch.matmul(A.float(), B.float().T)

        ctx.update({{"Model": Model, "get_inputs": get_inputs, "get_init_inputs": get_init_inputs}})

    exec({bundle.model_src!r}, ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")

    if use_mkb and execute_template is not None:
        device = torch.device("npu:0")
        synchronize = torch_npu.npu.synchronize
        correctness, info = execute_template(synchronize, device, ctx)
        if not correctness:
            print(info or "[FAIL] Output mismatch")
            return 1
        print("[eval] allclose=True [ref=MKB matmul_with_transposed_b]")
        return 0

    seed = 0
    trials = 3
    init_inputs = ctx["get_init_inputs"]()
    with torch.no_grad():
        set_seed(seed)
        original_model = ctx["Model"](*init_inputs).to("npu")
        _sync()
        set_seed(seed)
        custom_model = ctx["ModelNew"](*init_inputs).to("npu")
        _sync()
    with torch.no_grad():
        for _ in range(trials):
            inputs = [x.to("npu") if isinstance(x, torch.Tensor) else x for x in ctx["get_inputs"]()]
            _sync()
            ref_output = original_model(*inputs)
            _sync()
            new_output = custom_model(*inputs)
            _sync()
            if ref_output.shape != new_output.shape:
                print(f"[eval] shape mismatch: expected={{ref_output.shape}}, got={{new_output.shape}}")
                return 1
            if not torch.allclose(ref_output, new_output, atol=1e-2, rtol=1e-2):
                max_err = (ref_output - new_output).abs().max().item()
                print(f"[eval] allclose=False max_abs_err={{max_err}} [ref=custom small matmul]")
                return 1
    print("[eval] allclose=True [ref=custom small matmul]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
                (out_dir / "eval" / "spec.py").write_text(eval_py, encoding="utf-8")
            elif op_key == "layer_norm_custom":
                # Prefer MKB's reference implementation (to align with its dataset) when available.
                use_mkb_ref = True
                mkb_ref_path = (mkb_root / "mkb_reference" / "normalization" / "layer_norm.py").resolve()

                eval_py = f"""#!/usr/bin/env python3
import os
import sys
import pathlib
import torch
import torch_npu  # noqa: F401

def _sync():
    try:
        torch_npu.npu.synchronize()
    except Exception:
        pass

def set_seed(seed: int):
    torch.manual_seed(seed)

def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "{mod}")
    custom_ops = __import__(module_name)
    # Some bundles do `import custom_ops_lib` inside model_src. Force it to resolve
    # to the loaded extension module rather than an unrelated package.
    sys.modules["custom_ops_lib"] = custom_ops

    # ---- load reference + ModelNew ----
    ctx = {{"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}}

    if {use_mkb_ref!r}:
        mkb_root = pathlib.Path({str(mkb_root)!r})
        mkb_ref_path = pathlib.Path({str(mkb_ref_path)!r})
        # Import MKB correctness template directly from the original repo if present,
        # otherwise fall back to local simple check (this keeps the reference model aligned).
        mkb_repo = (pathlib.Path(__file__).resolve().parents[2] / "LLM" / "MultiKernelBench-main").resolve()
        if mkb_repo.exists():
            sys.path.insert(0, str(mkb_repo))
            from utils.correctness import execute_template  # type: ignore
        else:
            execute_template = None

        ref_src = mkb_ref_path.read_text(encoding="utf-8")
        exec(ref_src, ctx, ctx)
    else:
        # Fallback reference (kept for non-MKB bundles)
        normalized_shape = (1024,)
        def get_init_inputs():
            return [normalized_shape]
        def get_inputs():
            x = torch.randn(512, 1024, dtype=torch.float32)
            return [x]
        class Model(torch.nn.Module):
            def __init__(self, normalized_shape: tuple):
                super().__init__()
                self.ln = torch.nn.LayerNorm(normalized_shape, eps=1e-5)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ln(x)
        ctx.update({{"Model": Model, "get_inputs": get_inputs, "get_init_inputs": get_init_inputs}})

    # user-provided model_src (from txt bundle)
    exec({bundle.model_src!r}, ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")

    if {use_mkb_ref!r} and execute_template is not None:
        device = torch.device("npu:0")
        synchronize = torch_npu.npu.synchronize
        correctness, info = execute_template(synchronize, device, ctx)
        if not correctness:
            print(info or "[FAIL] Output mismatch")
            return 1
        print("[eval] allclose=True")
        return 0

    # fallback path: simple local check
    seed = 0
    trials = 3
    init_inputs = ctx["get_init_inputs"]()
    with torch.no_grad():
        set_seed(seed)
        original_model = ctx["Model"](*init_inputs).to("npu")
        _sync()
        set_seed(seed)
        custom_model = ctx["ModelNew"](*init_inputs).to("npu")
        _sync()
    with torch.no_grad():
        for _ in range(trials):
            inputs = [x.to("npu") if isinstance(x, torch.Tensor) else x for x in ctx["get_inputs"]()]
            _sync()
            ref_output = original_model(*inputs)
            _sync()
            new_output = custom_model(*inputs)
            _sync()
            if ref_output.shape != new_output.shape:
                print(f"[eval] shape mismatch: expected={{ref_output.shape}}, got={{new_output.shape}}")
                return 1
            if not torch.allclose(ref_output, new_output, atol=1e-4, rtol=1e-4):
                max_err = (ref_output - new_output).abs().max().item()
                print(f"[eval] allclose=False max_abs_err={{max_err}}")
                return 1
    print("[eval] allclose=True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
                (out_dir / "eval" / "spec.py").write_text(eval_py, encoding="utf-8")
            else:
                # Other .txt bundles with model_src: self-contained LayerNorm-shaped fallback (legacy).
                use_mkb_ref = False
                mkb_ref_path = (mkb_root / "mkb_reference" / "normalization" / "layer_norm.py").resolve()
                eval_py = f"""#!/usr/bin/env python3
import os
import sys
import pathlib
import torch
import torch_npu  # noqa: F401

def _sync():
    try:
        torch_npu.npu.synchronize()
    except Exception:
        pass

def set_seed(seed: int):
    torch.manual_seed(seed)

def main() -> int:
    module_name = os.environ.get("LLM4ASCENDC_OP_MODULE", "{mod}")
    custom_ops = __import__(module_name)
    sys.modules["custom_ops_lib"] = custom_ops

    ctx = {{"torch": torch, "torch_npu": torch_npu, "custom_ops_lib": custom_ops, "sys": sys}}

    normalized_shape = (1024,)
    def get_init_inputs():
        return [normalized_shape]
    def get_inputs():
        x = torch.randn(512, 1024, dtype=torch.float32)
        return [x]
    class Model(torch.nn.Module):
        def __init__(self, normalized_shape: tuple):
            super().__init__()
            self.ln = torch.nn.LayerNorm(normalized_shape, eps=1e-5)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ln(x)
    ctx.update({{"Model": Model, "get_inputs": get_inputs, "get_init_inputs": get_init_inputs}})

    exec({bundle.model_src!r}, ctx, ctx)
    if "ModelNew" not in ctx:
        raise RuntimeError("model_src must define ModelNew")

    seed = 0
    trials = 3
    init_inputs = ctx["get_init_inputs"]()
    with torch.no_grad():
        set_seed(seed)
        original_model = ctx["Model"](*init_inputs).to("npu")
        _sync()
        set_seed(seed)
        custom_model = ctx["ModelNew"](*init_inputs).to("npu")
        _sync()
    with torch.no_grad():
        for _ in range(trials):
            inputs = [x.to("npu") if isinstance(x, torch.Tensor) else x for x in ctx["get_inputs"]()]
            _sync()
            ref_output = original_model(*inputs)
            _sync()
            new_output = custom_model(*inputs)
            _sync()
            if ref_output.shape != new_output.shape:
                print(f"[eval] shape mismatch: expected={{ref_output.shape}}, got={{new_output.shape}}")
                return 1
            if not torch.allclose(ref_output, new_output, atol=1e-4, rtol=1e-4):
                max_err = (ref_output - new_output).abs().max().item()
                print(f"[eval] allclose=False max_abs_err={{max_err}}")
                return 1
    print("[eval] allclose=True")
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
    print("[smoke] no eval_src provided; skip correctness check")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
            (out_dir / "eval" / "spec.py").write_text(smoke, encoding="utf-8")

    # copy original txt for traceability
    (out_dir / "source_bundle.txt").write_text(txt_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return out_dir

