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


def gen_golden_data_simple():
    x1_gm_type = np.int8
    x2_gm_type = np.int8
    l0c_type = np.float32
    y_gm_type = np.int32

    used_core_num = 1
    m = 16
    n = 64
    k = 32
    single_core_m = 16
    single_core_n = 64
    single_core_k = 32
    base_m = 16
    base_n = 64
    base_k = 64
    depth_a1 = 1
    depth_b1 = 1
    step_m = 1
    step_n = 1
    step_ka = 1
    step_kb = 1
    a1_length = base_m * base_k * np.dtype(x1_gm_type).itemsize
    b1_length = base_k * base_n * np.dtype(x2_gm_type).itemsize
    co1_length = base_m * base_n * np.dtype(l0c_type).itemsize
    is_bias = 0
    trans_length = np.max([a1_length, b1_length, co1_length])
    iterate_order = 0
    x1_gm_shape = [16, 32]
    x2_gm_shape = [32, 64]
    y_gm_shape = [16, 64]
    bias_shape = [64]
    tiling_shape = [64]
    quant_shape = [64]
    workspace_shape = [16777]
    x1_gm = np.random.randint(1, 5, x1_gm_shape).astype(x1_gm_type)
    new_x1_gm = np.zeros(shape=[x1_gm_shape[0], x1_gm_shape[1] // 2]).astype(x1_gm_type)

    for i in range(x1_gm_shape[0]):
        for j in range(x1_gm_shape[1]):
            if j % 2 == 0:
                new_x1_gm[i][j // 2] = (x1_gm[i][j + 1] << 4) + (x1_gm[i][j] & 0x0f)

    x2_gm = np.random.randint(1, 5, x2_gm_shape).astype(x2_gm_type)
    new_x2_gm = np.zeros(shape=[x2_gm_shape[0], x2_gm_shape[1] // 2]).astype(x2_gm_type)

    for i in range(x2_gm_shape[0]):
        for j in range(x2_gm_shape[1]):
            if j % 2 == 0:
                new_x2_gm[i][j // 2] = (x2_gm[i][j + 1] << 4) + (x2_gm[i][j] & 0x0f)
    
    temp_quant_tensor = np.random.randint(1, 3, quant_shape).astype(np.float32)
    quant_tensor = np.frombuffer(temp_quant_tensor, np.int32)
    quant_tensor = quant_tensor.astype(np.int64)
    tiling = np.zeros((32), np.int32)

    l1_size = 0
    l0c_size = 0
    ub_size = 0
    # L1 Buffer 所需要的临时空间
    l1_size += depth_a1 * a1_length
    l1_size += depth_b1 * b1_length
    # 迭代次数
    iterate_size = 1
    # L0C Buffer 所需要的空间
    l0c_size += iterate_size * co1_length
    # 输入矩阵所需要的空间
    ub_size += iterate_size * trans_length
    # calBuf所需要的空间
    ub_size += trans_length
    # C矩阵所需要的空间
    ub_size += base_n * base_m * np.dtype(l0c_type).itemsize
    # tiling 参数
    tiling[0:32] = [used_core_num, m, n, k, k, single_core_m, single_core_n,
                    single_core_k, base_m, base_n, base_k,
                    depth_a1, depth_b1, step_m, step_n,
                    is_bias, trans_length, iterate_order, 0, l1_size, l0c_size,
                    ub_size, 1, 1, 1, 1, step_ka, step_kb, 2, 2, 1, 0]

    workspace = np.zeros(workspace_shape).astype(np.int32)
    bias = np.random.randint(1, 10, bias_shape).astype(np.int32)
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(y_gm_type)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    new_x1_gm.tofile("./input/input_x.bin")
    new_x2_gm.tofile("./input/input_y.bin")
    tiling.tofile("./input/input_tiling.bin")
    workspace.tofile("./input/input_workspace.bin")
    bias.tofile("./input/input_bias.bin")
    quant_tensor.tofile("./input/input_quant.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
