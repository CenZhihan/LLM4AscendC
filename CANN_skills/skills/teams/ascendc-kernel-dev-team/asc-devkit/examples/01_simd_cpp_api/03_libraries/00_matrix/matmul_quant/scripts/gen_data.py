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


import argparse
import os
import numpy as np


def gen_golden_data(quant_mode=0):
    m, n, k = 1024, 1024, 1024
    x1_gm = np.random.randint(-10, 10, [m, k]).astype(np.int8)
    x2_gm = np.random.randint(-10, 10, [k, n]).astype(np.int8)
    bias_gm = np.random.randint(-2, 2, [1, n]).astype(np.int32)

    y_gm_int32 = np.matmul(x1_gm.astype(np.int32), x2_gm.astype(np.int32)) + bias_gm
    golden = y_gm_int32.astype(np.float16)

    if quant_mode == 0:  # 0: scalar quant mode
        golden = golden * 0.1
    elif quant_mode == 1:  # 1: vector quant mode 
        quant_vector = np.random.uniform(0.1, 2.0, [1, n]).astype(np.float32)
        quant_vector_gm = np.frombuffer(quant_vector, np.int32)
        quant_vector_gm = quant_vector_gm.astype(np.uint64)

        quant_vector = quant_vector.view("uint32")
        for index, data in enumerate(quant_vector):
            # 1 sign bit, 8 exponent bits and 10 mantissa bits
            quant_vector[index] = np.bitwise_and(data, 0xFFFFE000)
        quant_vector = quant_vector.view("float32")
        for i in range(m):
            golden[i, :] = golden[i, :] * quant_vector

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    if quant_mode == 1:  
        quant_vector_gm.tofile("./input/quant_vector_gm.bin")
    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    gen_golden_data(args.m)
