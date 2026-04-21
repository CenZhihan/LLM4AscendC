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
    dtype = np.float32

    inputDy = np.random.randint(-10, 10, [4, 8, 512]).astype(dtype).astype(np.float32)
    res_for_gamma = np.random.randint(-10, 10, [4, 8, 512]).astype(dtype).astype(np.float32)

    reduce_axis = (0, 1)

    pd_beta = np.sum(inputDy, reduce_axis, keepdims=True)
    pd_gamma = np.sum(inputDy * res_for_gamma, reduce_axis, keepdims=True)

    inputDy = inputDy.astype(dtype)
    res_for_gamma = res_for_gamma.astype(dtype)

    os.makedirs("input", exist_ok=True)
    inputDy.tofile("./input/input_inputDy.bin")
    res_for_gamma.tofile("./input/input_resForGamma.bin")

    pd_gamma = pd_gamma.astype(dtype)
    pd_beta = pd_beta.astype(dtype)

    os.makedirs("output", exist_ok=True)
    pd_gamma.tofile("./output/golden_output_PdGamma.bin")
    pd_beta.tofile("./output/golden_output_PdBeta.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
