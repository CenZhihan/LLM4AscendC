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
    np.set_printoptions(threshold=sys.maxsize)
    shape = 32
    mode = 0
    count = 32
    dtype = np.float32
    is_dynamic = 0
    tiling = np.array([shape, count, mode, is_dynamic], dtype="uint32")

    src0 = np.random.uniform(10.0, 20.0, [32]).astype(dtype)
    src1 = np.random.uniform(0, 10.0, [32]).astype(dtype)
    condition = np.random.choice([True, False], size=shape)
    golden = np.zeros([32]).astype(dtype)

    if mode == 0:
        golden = np.where(condition, src0, src1)
    elif mode == 1:
        src0_val = src0[0]
        golden = np.where(condition, src0_val, src1)
    elif mode == 2:
        src1_val = src1[0]
        golden = np.where(condition, src0, src1_val)
    else:
        src0_val = src0[0]
        src1_val = src1[0]
        golden = np.where(condition, src0_val, src1_val)

    for i in range(shape - count):
        golden[count + i] = 0
    
    os.makedirs("input", exist_ok=True)
    src0.tofile("./input/input_src0.bin")
    src1.tofile("./input/input_src1.bin")
    condition.tofile("./input/input_condition.bin")
    tiling.tofile("./input/input_tiling.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
