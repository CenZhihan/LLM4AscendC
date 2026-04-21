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
import numpy as np


def gen_golden_data_simple():
    dtype = np.float32
    np.random.seed(123)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=32)

    first = 1
    last = 32
    src = np.random.uniform(0.0000000001, 100, [1, 32]).astype(dtype)

    reduce_axis = 1
    golden = np.mean(src, axis=reduce_axis)
    shape_bin = np.array([first, last], dtype="uint32")

    os.makedirs("input", exist_ok=True)
    src.tofile("./input/input_x.bin")
    shape_bin.tofile("./input/input_shape.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
