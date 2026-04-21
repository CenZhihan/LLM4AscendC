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
import enum
import numpy as np


class ScalarType(enum.Enum):
    scalarIndex0 = 1
    scalarIndex1 = 2
    tensorScalar = 3
    scalarTensor = 4

def gen_golden_data_simple():
    dtype = np.bool_
    np.set_printoptions(threshold=sys.maxsize, suppress=False)
    count = int(1024)
    data_size = int(1024)
    scalar_type = int(1)
    is_dynamic = int(0)
    src0 = np.random.uniform(-100, 100, [1024]).astype(dtype)
    src1 = np.random.uniform(-100, 100, [1024]).astype(dtype)
    golden = np.zeros([1024]).astype(dtype)

    if scalar_type == ScalarType.scalarIndex0.value or scalar_type == ScalarType.scalarTensor.value:
        golden[:count] = np.logical_and(src0[0], src1[:count]).astype(dtype)
    elif scalar_type == ScalarType.scalarIndex1.value or scalar_type == ScalarType.tensorScalar.value:
        golden[:count] = np.logical_and(src0[0], src1[:count]).astype(dtype)

    os.makedirs("input", exist_ok=True)
    src0.tofile("./input/input_src0.bin")
    src1.tofile("./input/input_src1.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
