"""
虚拟类别「test_set」：固定 12 个算子，用于组内统一评测/生成。

维护说明：只需修改下方 TEST_SET_RECORDS（增删改条目并保持与 vendor 数据集里的 op key 一致）。
顺序即为跑批与报告中的默认顺序。
"""
from __future__ import annotations

from typing import Any

# --- 唯一维护点：学长指定的 benchmark 集合（含原始 category / difficulty 备注）---
TEST_SET_RECORDS: tuple[dict[str, str], ...] = (
    {"op": "softplus", "category": "activation", "difficulty": "simple"},
    {"op": "log_softmax", "category": "activation", "difficulty": "complex"},
    {"op": "argmax_over_a_dimension", "category": "index", "difficulty": "medium"},
    {"op": "mean_reduction_over_a_dimension", "category": "reduce", "difficulty": "medium"},
    {"op": "masked_cumsum", "category": "math", "difficulty": "medium"},
    {"op": "layer_norm", "category": "normalization", "difficulty": "medium"},
    {"op": "max_pooling_3d", "category": "pooling", "difficulty": "medium"},
    {"op": "tall_skinny_matrix_multiplication", "category": "matmul", "difficulty": "medium"},
    {"op": "triplet_margin_loss", "category": "loss", "difficulty": "medium"},
    {"op": "vanilla_rnn_hidden", "category": "arch", "difficulty": "complex"},
    {
        "op": "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated",
        "category": "convolution",
        "difficulty": "complex",
    },
    {"op": "gemm_bias_add_hardtanh_mish_group_norm", "category": "fuse", "difficulty": "complex"},
)

assert len(TEST_SET_RECORDS) == 12

TEST_SET_OP_KEYS: tuple[str, ...] = tuple(r["op"] for r in TEST_SET_RECORDS)
assert len(set(TEST_SET_OP_KEYS)) == 12

TEST_SET_CATEGORY = "test_set"


def select_ops_by_categories(
    categories: list[str],
    dataset: dict[str, Any],
) -> tuple[list[str], bool]:
    """
    根据 CLI categories 解析要跑的算子列表。

    Returns:
        (ops, preserve_order)。若来自 test_set，preserve_order=True（保持 TEST_SET 定义顺序）；
        否则按字母序排序，preserve_order=False。
    """
    if categories == ["all"]:
        return sorted(dataset.keys()), False

    if TEST_SET_CATEGORY in categories:
        missing = [k for k in TEST_SET_OP_KEYS if k not in dataset]
        if missing:
            raise ValueError(
                f"test_set: 以下 key 不在当前数据集中: {missing}. "
                "请检查 vendor/mkb/dataset 或更新 generator/test_set_ops.py。"
            )
        return list(TEST_SET_OP_KEYS), True

    ops = [op for op in dataset if dataset[op]["category"] in categories]
    return sorted(ops), False
