"""
NPU Architecture Retriever for Ascend C kernel development agent.

Provides structured chip specification lookup based on the NPU architecture
documentation. Reads from Knowledge/arch/npu-arch-guide.md and maintains a
built-in chip specs database for quick lookup.
"""
import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Structured result types
# ============================================================

@dataclass
class ChipSpecResult:
    """Result of NPU chip specification lookup."""
    chip_name: str
    npu_arch: str                          # "DAV_2201" / "DAV_300" / ...
    ub_capacity_bytes: int
    vector_core_num: int
    cube_core_num: int
    hbm_capacity_gb: int
    max_block_dim: int
    max_tile_size: Dict[str, int]          # {dtype: max_elements}
    supported_apis: List[str]              # API categories supported
    features: List[str]                    # Feature list (Regbase, SIMT, FP8, etc.)
    arch_compile_macro: str                # DAV_2201 etc.
    soc_version: str                       # Ascend910B etc.
    details: str                           # Human-readable summary


# ============================================================
# Built-in chip specification database
#
# Source: Knowledge/arch/npu-arch-guide.md and Ascend C documentation.
# Values are approximate; for exact specs consult official Huawei docs.
# ============================================================

_CHIP_SPECS: Dict[str, ChipSpecResult] = {
    "Ascend910B": ChipSpecResult(
        chip_name="Ascend910B",
        npu_arch="DAV_2201",
        ub_capacity_bytes=196608,           # 192 KB
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=32,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline", "TQue", "TBuf"],
        arch_compile_macro="DAV_2201",
        soc_version="Ascend910B",
        details="Ascend910B 系列训练芯片，NpuArch=DAV_2201，通用 arch32 实现",
    ),
    "Ascend910B2": ChipSpecResult(
        chip_name="Ascend910B2",
        npu_arch="DAV_2201",
        ub_capacity_bytes=196608,
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=64,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline", "TQue", "TBuf"],
        arch_compile_macro="DAV_2201",
        soc_version="Ascend910B2",
        details="Ascend910B2 训练芯片，NpuArch=DAV_2201，与 910B 架构相同，HBM 更大",
    ),
    "Ascend910": ChipSpecResult(
        chip_name="Ascend910",
        npu_arch="DAV_1001",
        ub_capacity_bytes=196608,
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=32,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline"],
        arch_compile_macro="DAV_1001",
        soc_version="Ascend910",
        details="Ascend910 初代训练芯片，NpuArch=DAV_1001",
    ),
    "Ascend910_93": ChipSpecResult(
        chip_name="Ascend910_93",
        npu_arch="DAV_2201",
        ub_capacity_bytes=196608,
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=64,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline", "TQue", "TBuf"],
        arch_compile_macro="DAV_2201",
        soc_version="Ascend910_93",
        details="Ascend910_93 / Ascend910C 推理芯片，NpuArch=DAV_2201",
    ),
    "Ascend310P": ChipSpecResult(
        chip_name="Ascend310P",
        npu_arch="DAV_2002",
        ub_capacity_bytes=196608,
        vector_core_num=2,
        cube_core_num=1,
        hbm_capacity_gb=8,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline"],
        arch_compile_macro="DAV_2002",
        soc_version="Ascend310P",
        details="Ascend310P 推理芯片，NpuArch=DAV_2002",
    ),
    "Ascend310B": ChipSpecResult(
        chip_name="Ascend310B",
        npu_arch="DAV_3002",
        ub_capacity_bytes=196608,
        vector_core_num=2,
        cube_core_num=1,
        hbm_capacity_gb=8,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar"],
        features=["Double Buffer", "Pipeline"],
        arch_compile_macro="DAV_3002",
        soc_version="Ascend310B",
        details="Ascend310B 推理芯片，NpuArch=DAV_3002",
    ),
    "Ascend950DT": ChipSpecResult(
        chip_name="Ascend950DT",
        npu_arch="DAV_3510",
        ub_capacity_bytes=253952,           # 248 KB
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=64,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar", "SIMT"],
        features=["Double Buffer", "Pipeline", "TQue", "TBuf", "Regbase", "SIMT", "FP8 E5M2", "FP8 E4M3FN", "HiFloat8", "INT4"],
        arch_compile_macro="DAV_3510",
        soc_version="Ascend950DT",
        details="Ascend950DT 新一代 Decode 芯片，NpuArch=DAV_3510，支持 Regbase/SIMT/FP8",
    ),
    "Ascend950PR": ChipSpecResult(
        chip_name="Ascend950PR",
        npu_arch="DAV_3510",
        ub_capacity_bytes=253952,
        vector_core_num=4,
        cube_core_num=1,
        hbm_capacity_gb=64,
        max_block_dim=65535,
        max_tile_size={"float": 4096, "half": 8192, "int32": 4096, "int16": 8192, "int8": 16384},
        supported_apis=["Vector", "Cube", "Scalar", "SIMT"],
        features=["Double Buffer", "Pipeline", "TQue", "TBuf", "Regbase", "SIMT", "FP8 E5M2", "FP8 E4M3FN", "HiFloat8", "INT4"],
        arch_compile_macro="DAV_3510",
        soc_version="Ascend950PR",
        details="Ascend950PR 新一代 Prefill 芯片，NpuArch=DAV_3510，支持 Regbase/SIMT/FP8",
    ),
}

# Alias mapping: common alternate names -> canonical chip name
_CHIP_ALIASES: Dict[str, str] = {
    "910B": "Ascend910B",
    "910B2": "Ascend910B2",
    "910B1": "Ascend910B",
    "910B3": "Ascend910B",
    "910B4": "Ascend910B",
    "910": "Ascend910",
    "910_93": "Ascend910_93",
    "910C": "Ascend910_93",
    "310P": "Ascend310P",
    "310B": "Ascend310B",
    "950DT": "Ascend950DT",
    "950PR": "Ascend950PR",
    "950": "Ascend950DT",
}


# ============================================================
# Knowledge directory path
# ============================================================

def _get_knowledge_path() -> Optional[str]:
    """Get the Knowledge directory path."""
    # Try relative to this file
    this_dir = Path(__file__).parent
    knowledge = this_dir.parent / "Knowledge" / "arch"
    if knowledge.is_dir():
        return str(knowledge)
    return None


# ============================================================
# Main retriever class
# ============================================================

class NpuArchRetriever:
    """
    NPU Architecture specification lookup.

    Provides chip specification data from a built-in database and
    optionally from Knowledge/arch/npu-arch-guide.md.
    """

    def __init__(self):
        self._knowledge_path = _get_knowledge_path()

    def is_available(self) -> bool:
        """Check if chip spec database is available (always True for built-in)."""
        return True

    def lookup_chip_spec(self, chip_name: str) -> ChipSpecResult:
        """
        Look up chip specification by name.

        Args:
            chip_name: Chip name (e.g., "Ascend910B2", "910B", "Ascend950DT")

        Returns:
            ChipSpecResult with chip specifications
        """
        # Normalize input
        name = chip_name.strip()

        # Try exact match first
        if name in _CHIP_SPECS:
            return _CHIP_SPECS[name]

        # Try alias
        if name in _CHIP_ALIASES:
            return _CHIP_SPECS[_CHIP_ALIASES[name]]

        # Try case-insensitive match
        name_lower = name.lower()
        for key in _CHIP_SPECS:
            if key.lower() == name_lower:
                return _CHIP_SPECS[key]
        for alias, canonical in _CHIP_ALIASES.items():
            if alias.lower() == name_lower:
                return _CHIP_SPECS[canonical]

        # Fallback: return unknown chip with default values
        return ChipSpecResult(
            chip_name=chip_name,
            npu_arch="Unknown",
            ub_capacity_bytes=196608,
            vector_core_num=0,
            cube_core_num=0,
            hbm_capacity_gb=0,
            max_block_dim=0,
            max_tile_size={},
            supported_apis=[],
            features=[],
            arch_compile_macro="Unknown",
            soc_version=chip_name,
            details=f"未知芯片: {chip_name}。使用默认参数（UB=192KB）。请确认芯片型号是否正确。",
        )

    def list_chips(self) -> List[str]:
        """List all known chip names."""
        return sorted(set(list(_CHIP_SPECS.keys()) + list(_CHIP_ALIASES.keys())))

    def get_arch_guide_path(self) -> Optional[str]:
        """Get the path to the architecture guide document."""
        if self._knowledge_path:
            guide = os.path.join(self._knowledge_path, "npu-arch-guide.md")
            if os.path.isfile(guide):
                return guide
        return None
