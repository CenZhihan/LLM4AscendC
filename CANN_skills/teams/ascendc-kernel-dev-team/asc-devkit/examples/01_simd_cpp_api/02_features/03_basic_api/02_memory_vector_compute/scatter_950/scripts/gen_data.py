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
    input_type = np.dtype("float16")
    output_type = input_type
    type_size = input_type.itemsize
    block_length = 128 #输入一共是128个half类型的数

    input_shape = [block_length]
    output_shape = [block_length]
    input_x = np.arange(1, block_length + 1, dtype=input_type)
    input_y = np.arange((block_length - 1) * type_size, -1, -type_size, dtype=np.uint32)
    golden = np.zeros(output_shape).astype(output_type)
    for i in range(len(input_y)):
        golden[input_y[i] // type_size] = input_x[i]
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
