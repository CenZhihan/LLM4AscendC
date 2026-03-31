#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import numpy as np


def nd_to_nz_for_matrix_a(data_nd):
    """
    将ND格式的A矩阵转换为NZ格式
    A矩阵shape: [m, k], type:float16
    NZ格式: 每个BLOCK_CUBE x BLOCK_CUBE块内按列主序排列
    """
    m, k = data_nd.shape
    BLOCK_CUBE = 16
    
    # 计算对齐后的维度
    m_aligned = ((m + BLOCK_CUBE - 1) // BLOCK_CUBE) * BLOCK_CUBE
    k_aligned = ((k + BLOCK_CUBE - 1) // BLOCK_CUBE) * BLOCK_CUBE
    
    # 创建NZ格式数组（填充对齐）
    data_nz = np.zeros((m_aligned, k_aligned), dtype=data_nd.dtype)
    data_nz[:m, :k] = data_nd
    
    # 按BLOCK_CUBE x BLOCK_CUBE块重排数据
    for m_block in range(0, m_aligned, BLOCK_CUBE):
        for k_block in range(0, k_aligned, BLOCK_CUBE):
            block = data_nz[m_block:m_block+BLOCK_CUBE, k_block:k_block+BLOCK_CUBE]
            # NZ格式：块内按列主序排列
            block_flat = block.T.flatten()
            data_nz[m_block:m_block+BLOCK_CUBE, k_block:k_block+BLOCK_CUBE] = block_flat.reshape((BLOCK_CUBE, BLOCK_CUBE))
    
    return data_nz


def nd_to_nz_for_matrix_b(data_nd):
    """
    将ND格式的B矩阵转换为NZ格式
    B矩阵shape: [k, n], type:float16
    NZ格式: 每个BLOCK_CUBE x BLOCK_CUBE块内按行主序排列
    """
    k, n = data_nd.shape
    BLOCK_CUBE = 16
    
    # 计算对齐后的维度
    k_aligned = ((k + BLOCK_CUBE - 1) // BLOCK_CUBE) * BLOCK_CUBE
    n_aligned = ((n + BLOCK_CUBE - 1) // BLOCK_CUBE) * BLOCK_CUBE
    
    # 创建NZ格式数组（填充对齐）
    data_nz = np.zeros((k_aligned, n_aligned), dtype=data_nd.dtype)
    data_nz[:k, :n] = data_nd
    
    # 按BLOCK_CUBE x BLOCK_CUBE块重排数据
    for k_block in range(0, k_aligned, BLOCK_CUBE):
        for n_block in range(0, n_aligned, BLOCK_CUBE):
            block = data_nz[k_block:k_block+BLOCK_CUBE, n_block:n_block+BLOCK_CUBE]
            # NZ格式：块内按行主序排列（转置）
            block_flat = block.flatten()
            data_nz[k_block:k_block+BLOCK_CUBE, n_block:n_block+BLOCK_CUBE] = block_flat.reshape((BLOCK_CUBE, BLOCK_CUBE))
    
    return data_nz


def gen_golden_data_simple():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    m = 32
    n = 32
    k = 32

    # 生成ND格式的数据
    x1_nd = np.ones(m * k).reshape([m, k]).astype(np.float16)
    x2_nd = np.ones(k * n).reshape([k, n]).astype(np.float16)
    
    # 转换为NZ格式
    x1_nz = nd_to_nz_for_matrix_a(x1_nd)
    x2_nz = nd_to_nz_for_matrix_b(x2_nd)
    
    # 计算golden数据（使用ND格式计算）
    golden = (np.matmul(x1_nd.astype(np.float32), x2_nd.astype(np.float32))).astype(np.float32)
    
    # 保存NZ格式的输入数据
    x1_nz.tofile("./input/input_x.bin")
    x2_nz.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
