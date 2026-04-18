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


def split_and_reorder(data):
    data_uint8 = data.astype(np.uint8)
    low_bits_unsigned = data_uint8 & 0x0F
    high_bits_unsigned = (data_uint8 >> 4) & 0x0F
    low_bits_signed = np.where(low_bits_unsigned > 7, low_bits_unsigned - 16, low_bits_unsigned)
    high_bits_signed = np.where(high_bits_unsigned > 7, high_bits_unsigned - 16, high_bits_unsigned)
    result = np.empty(len(data) * 2, dtype=np.int8)
    result[0::2] = low_bits_signed
    result[1::2] = high_bits_signed
    return result

def gen_golden_data_simple():
    x1_gm_type = np.int8
    x2_gm_type = np.int8
    l0c_type = np.float32
    y_gm_type = np.int32
    used_core_num = 1
    m = 256
    n = 256
    k = 256
    single_core_m = 256
    single_core_n = 256
    single_core_k = 256
    base_m = 128
    base_n = 256
    base_k = 128
    input_format_a = 1
    input_format_b = 1
    output_format_c = 1
    depth_a1 = 4
    depth_b1 = 2
    step_m = 2
    step_n = 1
    step_ka = 2
    step_kb = 2
    a1_length = base_m * base_k * np.dtype(x1_gm_type).itemsize
    b1_length = base_k * base_n * np.dtype(x2_gm_type).itemsize
    co1_length = base_m * base_n * np.dtype(l0c_type).itemsize
    is_atomic_add = 0
    is_init_c = 0
    is_bias = 0
    quant_mode = 0
    trans_length = np.max([a1_length, b1_length, co1_length])
    iterate_order = 0
    is_transpose_a = 0
    is_transpose_b = 0
    x1_gm_int4x2 = np.random.randint(1, 10, [256, 128]).astype(np.int8)
    x2_gm_int4x2 = np.random.randint(1, 10, [256, 128]).astype(np.int8)
    x1_gm = np.random.randint(1, 10, [256, 256]).astype(x1_gm_type)
    x2_gm = np.random.randint(1, 10, [256, 256]).astype(x2_gm_type)
    for i in range(256):
        x1_gm[i] = split_and_reorder(x1_gm_int4x2[i])
        x2_gm[i] = split_and_reorder(x2_gm_int4x2[i])
    temp_quant_tensor = np.random.randint(1, 3, [256]).astype(np.float32)
    quant_tensor = np.frombuffer(temp_quant_tensor, np.int32)
    quant_tensor = quant_tensor.astype(np.uint64)
    tiling = np.zeros((32), np.int32)
    l1_size = 0
    l0c_size = 0
    ub_size = 0
    l1_size += depth_a1 * a1_length
    l1_size += depth_b1 * b1_length
    iterate_size = 1
    l0c_size += iterate_size * co1_length
    soc_version = 1
    ub_size += iterate_size * trans_length
    ub_size += trans_length
    ub_size += base_n * base_m * np.dtype(l0c_type).itemsize
    tiling[0:32] = [used_core_num, m, n, k, k,
                    single_core_m, single_core_n,
                    single_core_k, base_m, base_n,
                    base_k, depth_a1, depth_b1, step_m, step_n,
                    is_bias, trans_length, iterate_order, 0, l1_size,
                    l0c_size, ub_size, 1, 1, 1, 1, step_ka, step_kb, 1, 1, 1, 0]
    workspace = np.zeros([131072]).astype(np.float32)
    bias = np.random.randint(1, 10, [256]).astype(np.int32)
    golden = np.matmul(x1_gm.astype(np.int32), x2_gm.astype(np.int32)).astype(y_gm_type)

    x1_gm.fill(0)
    x2_gm.fill(0)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.tofile("./input/input_x.bin")
    x2_gm.tofile("./input/input_y.bin")
    tiling.tofile("./input/input_tiling.bin")
    workspace.tofile("./input/input_workspace.bin")
    bias.tofile("./input/input_bias.bin")
    quant_tensor.tofile("./input/input_quant.bin")
    golden.tofile("./output/golden.bin")
    x1_gm_int4x2.tofile("./input/input_x4.bin")
    x2_gm_int4x2.tofile("./input/input_y4.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
