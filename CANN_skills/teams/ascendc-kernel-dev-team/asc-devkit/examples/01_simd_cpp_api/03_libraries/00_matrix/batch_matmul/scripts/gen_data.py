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


def gen_golden_data():
    x1_gm_type = np.float16
    x2_gm_type = np.float16

    M = 192
    N = 1536
    K = 64

    aLayoutInfoB = 2
    aLayoutInfoS = 32
    aLayoutInfoN = 1
    aLayoutInfoG = 3
    aLayoutInfoD = 64
                
    bLayoutInfoB = 2
    bLayoutInfoS = 256
    bLayoutInfoN = 1
    bLayoutInfoG = 3
    bLayoutInfoD = 64

    x1_gm = np.random.randint(1, 10, [M, K]).astype(x1_gm_type)
    x2_gm = np.random.randint(1, 10, [K, N]).astype(x2_gm_type)
    # A_layout, B_layout, C_layout is BSNGD
    a_shape = [aLayoutInfoB, aLayoutInfoS, aLayoutInfoN, aLayoutInfoG, aLayoutInfoD]
    b_shape = [bLayoutInfoB, bLayoutInfoS, bLayoutInfoN, bLayoutInfoG, bLayoutInfoD]
    a = x1_gm.astype(np.float32).reshape(a_shape)
    b = x2_gm.astype(np.float32).reshape(b_shape)
    # a no_transpose, b transpose
    a_t = np.transpose(a, axes=(0, 2, 3, 1, 4))
    b_t = np.transpose(b, axes=(0, 2, 3, 4, 1))
    a_broadcast_shape = [max(aLayoutInfoB, bLayoutInfoB), max(aLayoutInfoN, bLayoutInfoN), max(aLayoutInfoG, bLayoutInfoG), aLayoutInfoS, aLayoutInfoD]
    b_broadcast_shape = [max(aLayoutInfoB, bLayoutInfoB), max(aLayoutInfoN, bLayoutInfoN), max(aLayoutInfoG, bLayoutInfoG), bLayoutInfoD, bLayoutInfoS]
    # print(" a brc shape is ", a_broadcast_shape, " b brc shape is ", b_broadcast_shape)
    a_broadcast = np.broadcast_to(a_t, a_broadcast_shape)
    b_broadcast = np.broadcast_to(b_t, b_broadcast_shape)
    golden = np.matmul(a_broadcast, b_broadcast).astype(np.float32)
    golden = np.transpose(golden, axes=(0, 3, 1, 2, 4))
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
    