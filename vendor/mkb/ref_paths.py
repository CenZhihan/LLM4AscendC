from __future__ import annotations

import pathlib

_PKG_DIR = pathlib.Path(__file__).resolve().parent
REFERENCE_ROOT = _PKG_DIR / "reference"


def get_ref_py_path(op_key: str) -> pathlib.Path:
    """
    Resolve vendored MKB reference source: reference/{category}/{op_key}.py
    """
    from vendor.mkb.dataset import dataset

    if op_key not in dataset:
        n = len(dataset)
        sample = ", ".join(sorted(dataset.keys())[:8])
        raise KeyError(
            f"Unknown MKB op_key {op_key!r} (from txt filename stem). "
            f"It must exist in vendored dataset ({n} ops). "
            f"Examples: {sample}, ..."
        )
    cat = dataset[op_key]["category"]
    p = REFERENCE_ROOT / cat / f"{op_key}.py"
    if not p.is_file():
        raise FileNotFoundError(f"Missing vendored reference file for op_key={op_key!r}: expected {p}")
    return p
