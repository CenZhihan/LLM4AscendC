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
import copy
import numpy as np


def gen_golden_data_simple():
    n, c, h, w, g = 2, 16, 8, 8, 4
    dtype = np.float32
    eps = 1e-5
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x = np.random.rand(n, c, h, w).astype(dtype)
    gamma = np.random.rand(1, c, 1, 1).astype(dtype)
    beta = np.random.rand(1, c, 1, 1).astype(dtype)
    input_x.tofile("./input/input_inputX.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    if dtype != np.float32 :
        input_x = input_x.astype(np.float32)
        gamma = gamma.astype(np.float32)
        beta = beta.astype(np.float32)
    input_x = input_x.reshape(n, g, c // g, h, w)
    mean = np.mean(input_x, (2, 3, 4), keepdims=True)
    var = np.var(input_x, (2, 3, 4), keepdims=True)
    input_x = (input_x - mean) / np.sqrt(var + eps)
    input_x = input_x.reshape(n, c, h, w)
    result = input_x * gamma + beta
    if dtype != np.float32:
        result = result.astype(dtype)
        mean = mean.astype(dtype)
        var = var.astype(dtype)
    result.tofile("./output/golden_output_result.bin")
    mean.tofile("./output/golden_output_mean.bin")
    var.tofile("./output/golden_output_variance.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
