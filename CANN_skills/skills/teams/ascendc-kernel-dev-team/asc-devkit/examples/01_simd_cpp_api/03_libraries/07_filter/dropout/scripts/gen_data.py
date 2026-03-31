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
import sys
from math import ceil
import numpy as np


def gen_golden_data_simple():

    src_shape = [4, 256]
    src_dtype = np.float32
    mask_shape = [4, 32]
    mask_dtype = np.uint8
    dst_shape = [4, 256]
    dst_dtype = np.float32
    tiling_shape = [1]
    tiling_dtype = np.float32

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    src = np.random.uniform(-100, 100, src_shape).astype(src_dtype)
    golden = np.zeros(dst_shape).astype(dst_dtype)
    prob = np.float32(np.random.rand())
    tiling = np.array(prob)

    if src_shape[1] == mask_shape[1]:
        mask = np.random.randint(0, 2, mask_shape).astype(mask_dtype)
        mask.tofile("./input/input_mask.bin")
        golden = src * mask
    elif src_shape[1] < mask_shape[1]:
        mask = np.random.randint(0, 2, mask_shape).astype(mask_dtype)
        mask.tofile("./input/input_mask.bin")
        mask = mask[:, :src_shape[1]]
        golden = src * mask
    elif ceil(src_shape[1] / 8) * 8 == mask_shape[1]:
        mask = np.random.randint(1, 255, mask_shape).astype(mask_dtype)
        mask.tofile("./input/input_mask.bin")
        result = []
        for i in range(mask_shape[0]):
            line = []
            for j in range(mask_shape[1]):
                value = bin(mask[i][j])[2:]
                value = list("0" * (8 - len(value)) + value)[::-1]
                line.extend(value)
            result.extend(line[:src_shape[1]])
        
        index = 0
        temp = np.zeros(params[0]).astype(mask_dtype)
        for i in range(src_shape[0]):
            for j in range(src_shape[1]):
                temp[i][j] = result[index]
                index += 1
        golden = src * temp
    else:
        mask = np.random.randint(1, 255, mask_shape).astype(mask_dtype)
        mask.tofile("./input/input_mask.bin")
        result = []
        for i in range(mask_shape[0]):
            line = []
            for j in range(mask_shape[1]):
                value = bin(mask[i][j])[2:]
                value = list("0" * (8 - len(value)) + value)[::-1]
                line.extend(value)
            result.extend(line[:src_shape[1]])

        index = 0
        temp = np.zeros(src_shape).astype(mask_dtype)
        for i in range(src_shape[0]):
            for j in range(src_shape[1]):
                temp[i][j] = result[index]
                index += 1
        golden = src * temp
    golden = golden * src_dtype(np.float32(1) / prob)
    golden = np.nan_to_num(golden)

    tiling.tofile("./input/input_tiling.bin")

    src.tofile("./input/input_src.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
