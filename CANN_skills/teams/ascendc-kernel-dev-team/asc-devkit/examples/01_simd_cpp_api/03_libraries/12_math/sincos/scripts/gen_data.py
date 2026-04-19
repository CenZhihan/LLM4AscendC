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
    count = int(1024)
    data_size = int(1024)
    np.set_printoptions(threshold=sys.maxsize)
    max_val = 10
    min_val = -10
    src = (np.random.uniform(min_val, max_val, data_size).astype(np.int32) * np.pi / 2).astype(dtype)
    golden0 = np.zeros([1024]).astype(dtype)
    golden1 = np.zeros([1024]).astype(dtype)
    golden0[:count] = np.sin(src[:count]).astype(dtype)
    golden1[:count] = np.cos(src[:count]).astype(dtype)
    os.makedirs("input", exist_ok=True)
    src.tofile("./input/input_src.bin")
    os.makedirs("output", exist_ok=True)
    golden0.tofile("./output/golden_0.bin")
    golden1.tofile("./output/golden_1.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
