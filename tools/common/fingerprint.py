#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Fingerprint:
    algo: str
    hexdigest: str
    files: list[str]

    def short(self) -> str:
        return self.hexdigest[:12]


def _hash_file(h: "hashlib._Hash", path: pathlib.Path) -> None:
    h.update(str(path).encode("utf-8"))
    h.update(b"\0")
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)


def compute_fingerprint(op_dir: pathlib.Path) -> Fingerprint:
    include: list[pathlib.Path] = []
    for rel in [
        "operator.json",
        "pybind/op.cpp",
        "eval/spec.py",
    ]:
        p = op_dir / rel
        if p.exists():
            include.append(p)

    for sub in ["op_host", "op_kernel"]:
        root = op_dir / sub
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                include.append(p)

    h = hashlib.sha256()
    for p in sorted(set(include), key=lambda x: str(x)):
        _hash_file(h, p)

    return Fingerprint(
        algo="sha256",
        hexdigest=h.hexdigest(),
        files=[str(p.relative_to(op_dir)) for p in sorted(set(include), key=lambda x: str(x))],
    )


def write_fingerprint(path: pathlib.Path, fp: Fingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"algo": fp.algo, "hexdigest": fp.hexdigest, "files": fp.files}, indent=2),
        encoding="utf-8",
    )


def read_fingerprint(path: pathlib.Path) -> Fingerprint | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return Fingerprint(algo=obj["algo"], hexdigest=obj["hexdigest"], files=list(obj.get("files", [])))

