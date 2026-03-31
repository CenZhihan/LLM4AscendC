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
    input_x_type = np.float32
    input_y_type = np.uint32
    output_type = input_x_type
    block_length = 128

    min_val, max_val = -1000, 1000
    input_shape = [block_length]
    output_shape = [block_length * 2]
    input_x = np.random.uniform(min_val, max_val, input_shape).astype(input_x_type)
    input_y = np.zeros(input_shape).astype(input_y_type)
    out = []
    for i in range(block_length // 32):
        base = i * 32
        sorted_pairs = sorted(zip(input_x[base : base + 32], input_y[base : base + 32]), reverse=True)
        out.extend([val for pair in sorted_pairs for val in pair])
    golden = np.array(out).astype(output_type)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
