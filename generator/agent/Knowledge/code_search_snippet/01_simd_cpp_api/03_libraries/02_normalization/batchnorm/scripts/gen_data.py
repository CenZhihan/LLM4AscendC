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


def reduce_add_data(origin_b, s_length, h_length, input_x):
    sh_length = s_length * h_length
    input_x = input_x.reshape([origin_b, sh_length]).astype(np.float32)
    res_list = np.zeros((1, sh_length)).astype(np.float32)
    for j in range(sh_length):
        res_sum = 0
        for i in range(origin_b):
            res_sum += input_x[i][j]
        res_list[0][j] = (res_sum).astype(np.float32)
    return res_list.reshape([s_length, h_length])


def gen_golden_data_simple():
    dtype = np.float32
    origin_b = 8
    epsilon = 0.01
    s_length = 8
    h_length = 8
    if (dtype == np.float16):
        epsilon = np.float16(epsilon)
    elif (dtype == np.float32):
        epsilon = np.float32(epsilon)
    input_x = np.arange(8 * 8 * 8).reshape([8, 8, 8]).astype(dtype)
    gamma = np.arange(8).astype(dtype).reshape([8, 1, 1])
    beta = np.arange(8).astype(dtype).reshape([8, 1, 1])
    os.makedirs("input", exist_ok=True)
    input_x.tofile("./input/input_inputX.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    if dtype != np.float32:
        input_x = input_x.astype(np.float32)
        gamma = gamma.astype(np.float32)
        beta = beta.astype(np.float32)
        epsilon = np.float32(epsilon)
    reduce_axis = 0
    vmuls_res = input_x / origin_b
    sh_length = input_x.shape[1] * input_x.shape[2]
    mean = reduce_add_data(origin_b, s_length, h_length, vmuls_res)
    mean_reshape = mean.reshape(s_length, h_length, 1)
    variance_vmuls_res = np.power((input_x - mean), 2) / origin_b
    variance = reduce_add_data(origin_b, s_length, h_length, variance_vmuls_res)
    vmuls_res = -0.5 * np.log(variance + epsilon)
    exp_res = np.exp(-0.5 * np.log(variance + epsilon))
    vmul_res = (input_x - mean) * exp_res
    vmul_res_gamma = gamma[:origin_b][:][:] * ((input_x - mean) * exp_res)
    result = gamma[:origin_b][:][:] * ((input_x - mean) * exp_res) + beta[:origin_b][:][:]
    if dtype != np.float32:
        result = result.astype(dtype)
        mean = mean.astype(dtype)
        variance = variance.astype(dtype)
        exp_res = exp_res.astype(dtype)
    os.makedirs("output", exist_ok=True)
    result.tofile("./output/golden_output_result.bin")
    mean.tofile("./output/golden_output_mean.bin")
    variance.tofile("./output/golden_output_variance.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
