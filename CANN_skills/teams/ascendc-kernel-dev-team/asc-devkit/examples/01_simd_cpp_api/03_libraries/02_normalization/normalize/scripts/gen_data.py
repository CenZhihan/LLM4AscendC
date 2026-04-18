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

import copy

A_SIZE = 8
R_SIZE = 64
R_SIZE_WITH_PAD = 64

def gen_golden_data_simple():
    inputX = np.random.uniform(1, 100, [A_SIZE, R_SIZE_WITH_PAD]).astype(np.float32)     # [A, R]
    meanX = np.random.uniform(1, 100, [A_SIZE]).astype(np.float32)                       # [A]
    varX = np.random.uniform(1, 100, [A_SIZE]).astype(np.float32)                        # [A]
    gamma = np.random.uniform(1, 100, [R_SIZE_WITH_PAD]).astype(np.float32)              # [R]
    beta = np.random.uniform(1, 100, [R_SIZE_WITH_PAD]).astype(np.float32)               # [R]
    output = inputX.copy()                                                             # [A, R]
    outputRstd = meanX.copy()                                                          # [A]

    # set padding part as 0
    for i in range(A_SIZE):
        for j in range(R_SIZE, R_SIZE_WITH_PAD):
            inputX[i][j] = np.float32(0)
    for j in range(R_SIZE, R_SIZE_WITH_PAD):
        beta[j] = np.float32(0)
        gamma[j] = np.float32(0)

    os.makedirs("input", exist_ok=True)
    inputX.tofile("./input/input_srcGm.bin")
    meanX.tofile("./input/input_inMeanGm.bin")
    varX.tofile("./input/input_inVarGm.bin")
    gamma.tofile("./input/input_inGammaGm.bin")
    beta.tofile("./input/input_inBetaGm.bin")

    epsilon = 0.001
    outputRstd = (1 / np.sqrt(varX + epsilon)).astype(np.float32)

    step1 = copy.deepcopy(inputX)
    step2 = copy.deepcopy(inputX)
    step3 = copy.deepcopy(inputX)
    result = copy.deepcopy(inputX)

    for i in range(A_SIZE):
        for j in range(R_SIZE_WITH_PAD):
            step1[i][j] = inputX[i][j] - meanX[i]

    for i in range(A_SIZE):
        for j in range(R_SIZE_WITH_PAD):
            step2[i][j] = step1[i][j] * outputRstd[i]

    for i in range(A_SIZE):
        for j in range(R_SIZE_WITH_PAD):
            step3[i][j] = step2[i][j] * gamma[j]

    for i in range(A_SIZE):
        for j in range(R_SIZE_WITH_PAD):
            if j < R_SIZE:
                result[i][j] = step3[i][j] + beta[j]
            else:
                result[i][j] = 0

    os.makedirs("output", exist_ok=True)
    result.tofile("./output/golden_outGm.bin")
    outputRstd.tofile("./output/golden_outRstdGm.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
