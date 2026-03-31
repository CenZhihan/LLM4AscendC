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


RN_SIZE = 1
AB_SIZE = 64
AB_LENGTH = 35
NREC = 1.0 / 8

def gen_golden_data_simple():
    x1 = np.random.uniform(1, 100, [RN_SIZE * AB_SIZE]).astype(np.float16)
    x2 = np.random.uniform(-60000, 60000, [RN_SIZE * AB_SIZE]).astype(np.float32)
    x3 = np.random.uniform(0, 60000, [RN_SIZE * AB_SIZE]).astype(np.float32)
    golden1 = x2.copy()
    golden2 = x3.copy()

    for i in range(AB_LENGTH):
        n = np.float32(NREC)
        golden1[i] = x2[i] + (x1[i] - x2[i]) * n
        golden2[i] = x3[i] + (x1[i] - x2[i]) * (x1[i] - golden1[i])

    os.makedirs("input", exist_ok=True)
    x1.tofile("./input/input_srcGm.bin")
    x2.tofile("./input/input_inMeanGm.bin")
    x3.tofile("./input/input_inVarGm.bin")
    os.makedirs("output", exist_ok=True)
    golden1.tofile("./output/golden_outMeanGm.bin")
    golden2.tofile("./output/golden_outVarGm.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
