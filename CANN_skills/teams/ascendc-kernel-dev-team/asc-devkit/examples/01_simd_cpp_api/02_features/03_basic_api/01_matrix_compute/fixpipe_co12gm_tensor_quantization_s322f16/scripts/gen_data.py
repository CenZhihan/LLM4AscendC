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


def clear_last_10bits_of_float32(f32_data):
    int32_data = f32_data.view("uint32")
    for index, data in enumerate(int32_data):
        int32_data[index] = np.bitwise_and(data, 0xFFFFE000)
    dealed_data = int32_data.view("float32")
    return dealed_data

def gen_golden_data():
    M = 32
    N = 32
    K = 32

    x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.int8)
    x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.int8)

    temp_deq_tensor_value = np.random.uniform(-2, 2, (N, )).astype("float32")
    deq_tensor_value = np.frombuffer(temp_deq_tensor_value, np.uint32)
    deq_tensor_value = deq_tensor_value.astype(np.uint64)

    temp_deq_tensor_value = deq_tensor_value.astype(np.int32)
    temp_deq_tensor_value = clear_last_10bits_of_float32(temp_deq_tensor_value)
    deq_tensor_value_fp32 = temp_deq_tensor_value.astype("float32")

    golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    golden_fp16 = golden.astype("float16")

    for i in range(M):
        for j in range(N):
            golden_fp16[i][j] = golden[i][j] * deq_tensor_value_fp32[j]

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    deq_tensor_value.tofile("./input/deq_gm.bin")
    golden_fp16.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()