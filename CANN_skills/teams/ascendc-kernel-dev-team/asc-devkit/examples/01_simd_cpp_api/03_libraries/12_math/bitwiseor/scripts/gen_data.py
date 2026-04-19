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
    dtype = np.int32
    src0_shape = [1024]
    src1_shape = [1024]
    dst_shape = [1024]

    np.set_printoptions(threshold=sys.maxsize, suppress=False)

    src0 = np.random.uniform(-1000, 1000, src0_shape).astype(dtype)
    src1 = np.random.uniform(-1000, 1000, src1_shape).astype(dtype)
    golden = np.zeros(dst_shape).astype(dtype)

    golden[:1023] = np.bitwise_or(src0[:1023], src1[:1023])

    os.makedirs("input", exist_ok=True)
    src0.tofile("./input/input_src0.bin")
    src1.tofile("./input/input_src1.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
