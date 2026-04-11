#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class KernelbenchOp:
    project_name: str
    op_key: str
    dir_path: pathlib.Path
    project_json_path: pathlib.Path
    host_tiling_h: pathlib.Path
    host_cpp: pathlib.Path
    kernel_cpp: pathlib.Path
    pybind_cpp: pathlib.Path
    api_func_name: str


def _read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _strip_custom_suffix(project_name: str) -> str:
    return project_name[:-6] if project_name.endswith("Custom") else project_name


def _resolve_mkb_op_key(project_name: str, *, dataset_keys: set[str]) -> str | None:
    """
    Map kernelbench project name -> MKB op_key (txt filename stem).

    Primary: CamelCase -> snake_case.
    Fallback: insert '_' before dimension suffixes like 'pooling1d' -> 'pooling_1d'
              but keep 'conv2d' as-is (MKB uses 'conv2d_*' not 'conv_2d_*').
    """
    base = _camel_to_snake(_strip_custom_suffix(project_name))
    manual = {
        # Kernelbench naming vs MKB op_key naming differences.
        "HardSigmoidCustom": "hardsigmoid",
        "HardTanhCustom": "hardtanh",
        "LeNet5Custom": "lenet5",
        "GruBidirectionalCustom": "gru_birectional",  # note: MKB key has this spelling
        "GemmMultiplyLeakyReluCustom": "gemm_multiply_leakyrelu",
        "GemmScalingHardtanhGeluCustom": "gemm_scaling_hard_tanh_gelu",
        "MatmulDropoutSoftmaxCustom": "matmul_dropout_mean_softmax",
        "ConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom": "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply",
        "ConvTranspose2dSoftmaxBiasAddScalingSigmoidCustom": "convtranspose2d_softmax_biasadd_scaling_sigmoid",
        "ConvTranspose3dLogSumExpHardSwishSubtractClampCustom": "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max",
    }
    if project_name in manual and manual[project_name] in dataset_keys:
        return manual[project_name]
    if base in dataset_keys:
        return base

    # Only add underscore before '<token><Nd>' when the token is preceded by '_'.
    # Examples: average_pooling1d -> average_pooling_1d, conv_standard1d -> conv_standard_1d,
    # but conv2d_* stays conv2d_* (no preceding underscore).
    # Handle both "...token2d_" and "...token2d$" forms.
    alt = re.sub(r"(?<=_)([a-z]+)(\d+d)(?=_)", r"\1_\2", base)
    alt = re.sub(r"(?<=_)([a-z]+)(\d+d)$", r"\1_\2", alt)
    if alt in dataset_keys:
        return alt

    # Sometimes multiple tokens can contain Nd, apply iteratively until stable.
    prev = alt
    while True:
        nxt = re.sub(r"(?<=_)([a-z]+)(\d+d)(?=_)", r"\1_\2", prev)
        nxt = re.sub(r"(?<=_)([a-z]+)(\d+d)$", r"\1_\2", nxt)
        if nxt == prev:
            break
        prev = nxt
        if prev in dataset_keys:
            return prev

    return None


_IMPL_RE = re.compile(r'\bm\.impl\(\s*"(?P<name>[^"]+)"\s*,')


def _infer_api_func_name(pybind_cpp_src: str) -> str:
    """
    Choose the primary exported op API name used by custom_ops_lib.
    Prefer the first m.impl("xxx", ...) not ending with "_out".
    """
    names = [m.group("name") for m in _IMPL_RE.finditer(pybind_cpp_src)]
    if not names:
        raise ValueError("cannot find TORCH_LIBRARY_IMPL m.impl(\"...\") in pybind op.cpp")
    for n in names:
        if not n.endswith("_out"):
            return n
    return names[0]


def _find_single(globbed: list[pathlib.Path], *, what: str, base_dir: pathlib.Path) -> pathlib.Path:
    if not globbed:
        raise FileNotFoundError(f"{base_dir}: missing {what}")
    if len(globbed) > 1:
        # Prefer the shortest / most direct hit.
        globbed = sorted(globbed, key=lambda p: (len(p.parts), str(p)))
    return globbed[0]


def _discover_one(project_dir: pathlib.Path, project_name: str, op_key: str) -> KernelbenchOp:
    # project json
    project_json_path = _find_single(
        [p for p in project_dir.glob("*.json") if p.name not in ("result.json", "manifest.json")],
        what="project json (*.json, excluding result.json)",
        base_dir=project_dir,
    )

    # host sources
    host_dir = project_dir / "op_host"
    host_tiling_h = _find_single(list(host_dir.glob("*_tiling.h")), what="op_host/*_tiling.h", base_dir=host_dir)
    host_cpp = _find_single(
        [p for p in host_dir.glob("*.cpp") if p.name != "op_proto.cc"],
        what="op_host/*.cpp",
        base_dir=host_dir,
    )

    # kernel sources
    kernel_dir = project_dir / "op_kernel"
    kernel_cpp = _find_single(list(kernel_dir.glob("*.cpp")), what="op_kernel/*.cpp", base_dir=kernel_dir)

    # pybind
    pybind_cpp = project_dir / "CppExtension" / "csrc" / "op.cpp"
    if not pybind_cpp.is_file():
        raise FileNotFoundError(pybind_cpp)
    api_func_name = _infer_api_func_name(_read_text(pybind_cpp))

    return KernelbenchOp(
        project_name=project_name,
        op_key=op_key,
        dir_path=project_dir,
        project_json_path=project_json_path,
        host_tiling_h=host_tiling_h,
        host_cpp=host_cpp,
        kernel_cpp=kernel_cpp,
        pybind_cpp=pybind_cpp,
        api_func_name=api_func_name,
    )


def write_txt_bundle(op: KernelbenchOp, *, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{op.op_key}.txt"

    project_json_src = _read_text(op.project_json_path)
    host_tiling_src = _read_text(op.host_tiling_h)
    host_operator_src = _read_text(op.host_cpp)
    kernel_src = _read_text(op.kernel_cpp)
    python_bind_src = _read_text(op.pybind_cpp)

    # Generic ModelNew: accept any number of inputs from MKB get_inputs()
    model_src = f"""import torch
import torch_npu
import custom_ops_lib

_FN = {op.api_func_name!r}

class ModelNew(torch.nn.Module):
    def __init__(self, *init_inputs):
        super(ModelNew, self).__init__()

    def forward(self, *inputs):
        fn = getattr(custom_ops_lib, _FN)
        return fn(*inputs)
"""

    payload = (
        "```python\n"
        f"project_json_src='''\n{project_json_src}'''\n\n"
        f'host_tiling_src="""\n{host_tiling_src}"""\n\n'
        f'host_operator_src="""\n{host_operator_src}"""\n\n'
        f'kernel_src="""\n{kernel_src}"""\n\n'
        f'python_bind_src="""\n{python_bind_src}"""\n\n'
        f"model_src='''\n{model_src}'''\n"
        "```\n"
    )
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def main() -> int:
    kb_root = pathlib.Path("/root/LLM/ops-kernelbench-910b")
    manifest_path = kb_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    from vendor.mkb.dataset import dataset

    dataset_keys = set(dataset.keys())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernels") or []
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("manifest.json: missing kernels list")

    out_dir = ROOT / "output" / "kernelbench165_txt"
    # do not delete user data elsewhere; only clear our generated dir
    if out_dir.exists():
        for p in out_dir.glob("*.txt"):
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for item in kernels:
        project_name = str(item["kernel_project_name"])
        op_key = _resolve_mkb_op_key(project_name, dataset_keys=dataset_keys)
        if op_key is None:
            skipped.append((project_name, "cannot map to vendored MKB op_key"))
            continue

        project_dir = kb_root / project_name
        try:
            op = _discover_one(project_dir, project_name, op_key)
            write_txt_bundle(op, out_dir=out_dir)
            ok += 1
        except Exception as e:
            failed.append((project_name, f"{type(e).__name__}: {e}"))

    print(f"[kernelbench] generated txt: {ok} / {len(kernels)} -> {out_dir}")
    if skipped:
        print(f"[kernelbench] skipped (op_key not in dataset): {len(skipped)}")
        for name, why in skipped[:20]:
            print(f"  - {name}: {why}")
        if len(skipped) > 20:
            print("  ...")
    if failed:
        print(f"[kernelbench] failed: {len(failed)}")
        for name, why in failed[:20]:
            print(f"  - {name}: {why}")
        if len(failed) > 20:
            print("  ...")
    # Hard fail if we cannot generate full set; this protects batch eval.
    return 0 if (not skipped and not failed and ok == len(kernels)) else 2


if __name__ == "__main__":
    raise SystemExit(main())

