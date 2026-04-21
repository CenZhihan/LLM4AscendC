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
    m, n, k = 32, 32, 32
    quant_val = 2.0

    x1_gm = np.random.uniform(0, 10, [m, k]).astype(np.int8)
    x2_gm = np.random.uniform(0, 10, [k, n]).astype(np.int8)
    golden = (np.matmul(x1_gm.astype(np.int32), x2_gm.astype(np.int32), dtype=np.int32)).astype(np.float16)
    golden = golden * quant_val

    golden_nz = golden.reshape((int(m / 16), 16, int(n / 16), 16)).transpose(2, 0, 1, 3)

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden_nz.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()