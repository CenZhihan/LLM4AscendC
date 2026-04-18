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
from enum import Enum


class CMPMODE(Enum):
    EQ = "EQ"
    NE = "NE"
    GE = "GE"
    LE = "LE"
    GT = "GT"
    LT = "LT"


def get_range_by_dtype(input_type):
    try:
        if input_type == np.float16 or input_type == np.float32 or input_type == np.float64:
            return np.finfo(input_type).min, np.finfo(input_type).max
        else:
            return np.iinfo(input_type).min, np.iinfo(input_type).max
    except ValueError:
        print(f"Unsupported data type:{input_type}")

def gen_golden_data_simple():
    compare_mode = CMPMODE.LT
    input_type = np.float32
    output_type = np.uint8
    use_core_num = 1
    block_length = 64
    min_val, max_val = get_range_by_dtype(input_type)
    input_shape = [use_core_num, 64]
    output_shape = [use_core_num, 32]
    input_x = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    input_y = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    golden = np.zeros(output_shape).astype(output_type)
    #逐元素对src0和src1中的数据进行比较，如果src0中的元素小于src1中的元素，dst结果中对应的比特位置1；反之，则置0。
    if compare_mode == CMPMODE.EQ:
        compare_result = (input_x == input_y).astype(output_type)
    elif compare_mode == CMPMODE.NE:
        compare_result = (input_x != input_y).astype(output_type)
    elif compare_mode == CMPMODE.LE:
        compare_result = (input_x <= input_y).astype(output_type)
    elif compare_mode == CMPMODE.LT:
        compare_result = (input_x < input_y).astype(output_type)
    elif compare_mode == CMPMODE.GT:
        compare_result = (input_x > input_y).astype(output_type)
    elif compare_mode == CMPMODE.GE:
        compare_result = (input_x >= input_y).astype(output_type)
    for i in range(use_core_num):
        for j in range(block_length // 8):
            bits = compare_result[i, j * 8 : (j + 1) * 8]
            byte_val = 0
            for k in range(8):
                if bits[k] == 1:
                    byte_val |= (1<<k)
            golden[i, j] = byte_val

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
