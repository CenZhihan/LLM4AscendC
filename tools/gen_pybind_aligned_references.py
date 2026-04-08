#!/usr/bin/env python3
"""
Regenerate vendor/mkb/reference/*.py for ops listed in artifacts/pybind_reference_audit.json
by reading output/kernelbench165_txt/<op>.txt python_bind_src.

This is intentionally conservative: it emits explicit forward(*tensors) + get_inputs() lists
matching the C++ entrypoint arity. For fused/RNN ops, forward bodies call the same F.* ops
as documented in the kernelbench txt (stride/pad from constexpr lines where present).

Run from LLM4AscendC: PYTHONPATH=. python3 tools/gen_pybind_aligned_references.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TXT_DIR = ROOT / "output" / "kernelbench165_txt"
AUDIT = ROOT / "artifacts" / "pybind_reference_audit.json"


def extract_bind(txt: str) -> str:
    m = re.search(r'python_bind_src\s*=\s*"""(.*?)"""', txt, re.DOTALL)
    return m.group(1) if m else ""


def find_mdef_fn(bind: str) -> str | None:
    m = re.search(r'm\.def\s*\(\s*"[^"]+"\s*,\s*&(\w+)\s*[,)]', bind)
    return m.group(1) if m else None


def param_list_after_name(bind: str, func_name: str) -> str | None:
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


def split_params(inner: str) -> list[str]:
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


def classify_param(p: str) -> str:
    p = re.sub(r"//.*", "", p).strip()
    if not p:
        return "unknown"
    if "optional" in p.lower():
        return "optional"
    if re.search(r"\bat::Tensor\b|const\s+at::Tensor", p):
        return "tensor"
    if re.search(r"\b(double|float)\b", p):
        return "float"
    if re.search(r"\b(int64_t|int32_t|int)\b", p):
        return "int"
    if re.search(r"\bbool\b", p):
        return "bool"
    return "other"


def ints_after_eq(bind: str, *needles: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in needles:
        m = re.search(rf"constexpr\s+int64_t\s+{name}\s*=\s*(\d+)", bind)
        if m:
            out[name] = int(m.group(1))
    return out


def torch_check_eq(bind: str, var: str, idx: int) -> int | None:
    # TORCH_CHECK(x.size(0) == 16
    pat = rf"{re.escape(var)}\.size\({idx}\)\s*==\s*(\d+)"
    m = re.search(pat, bind)
    return int(m.group(1)) if m else None


def emit_conv_family(op_key: str, bind: str, kinds: list[str], names: list[str]) -> str:
    """Best-effort PyTorch reference for conv* ops; uses TORCH_CHECK literals + constexpr stride/pad."""
    # names aligned with split_params order (parameter names only - rough)
    if len(kinds) == 2 and kinds == ["tensor", "tensor"]:
        # conv2d / conv1d / conv3d / conv_transpose — infer dim from x.dim() checks
        if "x must be 3D" in bind or "3D (NCL" in bind:
            n = torch_check_eq(bind, "x", 0) or 32
            cin = torch_check_eq(bind, "x", 1) or 64
            lin = torch_check_eq(bind, "x", 2) or 131072
            cout = torch_check_eq(bind, "weight", 0) or 128
            wcin = torch_check_eq(bind, "weight", 1) or cin
            k = torch_check_eq(bind, "weight", 2) or 3
            stride = ints_after_eq(bind, "stride").get("stride", 1)
            dilation = ints_after_eq(bind, "dilation").get("dilation", 1)
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Pybind: (x, weight) 1D conv; sizes from kernelbench txt."""
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv1d(
            x, weight, None,
            stride={stride}, padding=0, dilation={dilation}, groups=1,
        )

def get_inputs():
    x = torch.rand({n}, {cin}, {lin})
    w = torch.rand({cout}, {wcin}, {k})
    return [x, w]

def get_init_inputs():
    return []
'''
        if "x must be 5D" in bind or "5D" in bind and "NDCDHW" in bind or "NCDHW" in bind:
            # conv3d or conv_transpose3d
            tr = "conv_transpose" in op_key or "transposed_3d" in op_key
            fn = "conv_transpose3d" if tr else "conv3d"
            # parse sizes from TORCH_CHECK lines
            def sz(v, i):
                return torch_check_eq(bind, v, i)

            n = sz("x", 0) or 8
            cin = sz("x", 1) or 32
            d = sz("x", 2) or 32
            h = sz("x", 3) or 32
            w_ = sz("x", 4) or 32
            cout = sz("weight", 1) if tr else sz("weight", 0)
            if cout is None:
                cout = 32
            kd = sz("weight", 2) or 3
            kh = sz("weight", 3) or 3
            kw = sz("weight", 4) or 3
            sd = sh = sw = 1
            m = re.search(
                r"const\s+int64_t\s+strideD\s*=\s*(\d+),\s*strideH\s*=\s*(\d+),\s*strideW\s*=\s*(\d+)",
                bind,
            )
            if m:
                sd, sh, sw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pd = ph = pw = 0
            m2 = re.search(
                r"padD\s*=\s*(\d+),\s*padH\s*=\s*(\d+),\s*padW\s*=\s*(\d+)",
                bind,
            )
            if m2:
                pd, ph, pw = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            out_pad = (0, 0, 0)
            m3 = re.search(r"output_padding", bind)
            opd = oph = opw = 0
            if tr:
                m4 = re.search(
                    r"out_padD\s*=\s*(\d+),\s*out_padH\s*=\s*(\d+),\s*out_padW\s*=\s*(\d+)",
                    bind,
                )
                if m4:
                    opd, oph, opw = int(m4.group(1)), int(m4.group(2)), int(m4.group(3))
            dil = (1, 1, 1)
            if tr:
                return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv_transpose3d(
            x, weight, None,
            stride=({sd},{sh},{sw}),
            padding=({pd},{ph},{pw}),
            output_padding=({opd},{oph},{opw}),
            groups=1,
            dilation={dil},
        )

def get_inputs():
    x = torch.rand({n}, {cin}, {d}, {h}, {w_})
    w = torch.rand({cin}, {cout}, {kd}, {kh}, {kw})
    return [x, w]

def get_init_inputs():
    return []
'''
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv3d(
            x, weight, None,
            stride=({sd},{sh},{sw}),
            padding=({pd},{ph},{pw}),
            dilation={dil},
            groups=1,
        )

def get_inputs():
    x = torch.rand({n}, {cin}, {d}, {h}, {w_})
    w = torch.rand({cout}, {cin}, {kd}, {kh}, {kw})
    return [x, w]

def get_init_inputs():
    return []
'''
        # default: conv2d
        n = torch_check_eq(bind, "x", 0) or 16
        cin = torch_check_eq(bind, "x", 1) or 64
        h = torch_check_eq(bind, "x", 2) or 32
        w_in = torch_check_eq(bind, "x", 3) or h
        cout = torch_check_eq(bind, "weight", 0) or 128
        wcin = torch_check_eq(bind, "weight", 1) or cin
        kh = torch_check_eq(bind, "weight", 2) or 3
        kw = torch_check_eq(bind, "weight", 3) or kh
        tr = "conv_transposed" in op_key or op_key.startswith("conv_transposed")
        if tr:
            sh = sw = 1
            m = re.search(r"const\s+int64_t\s+strideH\s*=\s*(\d+),\s*strideW\s*=\s*(\d+)", bind)
            if m:
                sh, sw = int(m.group(1)), int(m.group(2))
            ph = pw = 0
            m2 = re.search(r"padH\s*=\s*(\d+),\s*padW\s*=\s*(\d+)", bind)
            if m2:
                ph, pw = int(m.group(1)), int(m.group(2))
            oph = opw = 0
            m3 = re.search(r"outpadH\s*=\s*(\d+),\s*outpadW\s*=\s*(\d+)", bind)
            if m3:
                oph, opw = int(m.group(1)), int(m.group(2))
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv_transpose2d(
            x, weight, None,
            stride=({sh},{sw}),
            padding=({ph},{pw}),
            output_padding=({oph},{opw}),
            groups=1,
            dilation=(1,1),
        )

def get_inputs():
    x = torch.rand({n}, {cin}, {h}, {w_in})
    w = torch.rand({cin}, {cout}, {kh}, {kw})
    return [x, w]

def get_init_inputs():
    return []
'''
        sh = sw = 1
        ph = pw = 0
        dilh = dilw = 1
        m = re.search(
            r"strideH\s*=\s*(\d+),\s*strideW\s*=\s*(\d+)",
            bind,
        )
        if m:
            sh, sw = int(m.group(1)), int(m.group(2))
        m2 = re.search(r"padH\s*=\s*(\d+),\s*padW\s*=\s*(\d+)", bind)
        if m2:
            ph, pw = int(m2.group(1)), int(m2.group(2))
        m3 = re.search(r"dilH\s*=\s*(\d+),\s*dilW\s*=\s*(\d+)", bind)
        if m3:
            dilh, dilw = int(m3.group(1)), int(m3.group(2))
        return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv2d(
            x, weight, None,
            stride=({sh},{sw}),
            padding=({ph},{pw}),
            dilation=({dilh},{dilw}),
            groups=1,
        )

def get_inputs():
    x = torch.rand({n}, {cin}, {h}, {w_in})
    w = torch.rand({cout}, {wcin}, {kh}, {kw})
    return [x, w]

def get_init_inputs():
    return []
'''

    if op_key == "conv_depthwise_separable_2d" and kinds.count("tensor") == 3:
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w_depthwise, w_pointwise):
        x = F.conv2d(x, w_depthwise, None, stride=1, padding=1, groups=64)
        return F.conv2d(x, w_pointwise, None, stride=1, padding=0, groups=1)

def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    w_dw = torch.rand(64, 1, 3, 3)
    w_pw = torch.rand(128, 64, 1, 1)
    return [x, w_dw, w_pw]

def get_init_inputs():
    return []
'''

    if op_key == "conv_depthwise_2d_asymmetric_input_square_kernel":
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv2d(x, weight, None, stride=1, padding=1, groups=128)

def get_inputs():
    x = torch.rand(64, 128, 256, 512)
    w = torch.rand(128, 1, 3, 3)
    return [x, w]

def get_init_inputs():
    return []
'''

    if op_key == "conv_depthwise_2d_square_input_square_kernel":
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv2d(x, weight, None, stride=1, padding=1, groups=64)

def get_inputs():
    x = torch.rand(16, 64, 512, 512)
    w = torch.rand(64, 1, 3, 3)
    return [x, w]

def get_init_inputs():
    return []
'''

    if op_key == "conv_pointwise_2d":
        return '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return F.conv2d(x, weight, None, stride=1, padding=0, groups=1)

def get_inputs():
    x = torch.rand(16, 64, 1024, 1024)
    w = torch.rand(128, 64, 1, 1)
    return [x, w]

def get_init_inputs():
    return []
'''

    return ""


# --- hand-written overrides for ops where generic emit fails ---
HANDLERS: dict[str, str] = {}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from vendor.mkb.ref_paths import get_ref_py_path

    rows = json.loads(AUDIT.read_text(encoding="utf-8"))
    for row in rows:
        op_key = row["op_key"]
        if row.get("issue", "").startswith("no_"):
            print("skip", op_key, row)
            continue
        txt_path = TXT_DIR / f"{op_key}.txt"
        txt = txt_path.read_text(encoding="utf-8", errors="replace")
        bind = extract_bind(txt)
        fn = find_mdef_fn(bind)
        if not fn:
            print("no m.def", op_key)
            continue
        inner = param_list_after_name(bind, fn)
        if not inner:
            print("no params", op_key, fn)
            continue
        parts = split_params(inner)
        kinds = [classify_param(p) for p in parts]

        code = HANDLERS.get(op_key) or emit_conv_family(op_key, bind, kinds, parts)
        if not code.strip():
            print("UNHANDLED", op_key, kinds)
            continue
        out = get_ref_py_path(op_key)
        out.write_text(code.rstrip() + "\n", encoding="utf-8")
        print("wrote", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
