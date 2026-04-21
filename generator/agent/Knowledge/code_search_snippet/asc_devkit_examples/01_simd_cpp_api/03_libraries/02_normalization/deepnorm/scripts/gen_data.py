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
    input_x = np.arange(4 * 16 * 64).reshape([4, 16, 64]).astype(dtype)
    input_gx = np.arange(4 * 16 * 64).reshape([4, 16, 64]).astype(dtype)
    gamma = np.arange(64).reshape([64]).astype(dtype)
    beta = np.arange(64).reshape([64]).astype(dtype)
    alpha_dtype = dtype
    alpha = alpha_dtype(-5.5)
    epsilon = alpha_dtype(65504)
    minNum, maxNum = alpha_dtype(65500), (65504)
    os.makedirs("input", exist_ok=True)
    input_x.tofile("./input/input_inputX.bin")
    input_gx.tofile("./input/input_inputGx.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    if dtype != np.float32:
        input_x = input_x.astype(np.float32)
        input_gx = input_gx.astype(np.float32)
        gamma = gamma.astype(np.float32)
        beta = beta.astype(np.float32)
        alpha = alpha.astype(np.float32)
        epsilon = np.float32(epsilon)
    mulsX = alpha * input_x
    X1 = mulsX + input_gx
    reduce_axis = 2
    mean = np.mean(X1, reduce_axis, keepdims=True)
    variance = np.mean(np.power((X1 - mean), 2), reduce_axis, keepdims=True)
    result = gamma * ((X1 - mean) / np.sqrt(variance + epsilon)) + beta
    if dtype != np.float32:
        result = result.astype(dtype)
        mean = mean.astype(dtype)
        variance = variance.astype(dtype)
    os.makedirs("output", exist_ok=True)
    result.tofile("./output/golden_output_result.bin")
    mean.tofile("./output/golden_output_mean.bin")
    variance.tofile("./output/golden_output_variance.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
