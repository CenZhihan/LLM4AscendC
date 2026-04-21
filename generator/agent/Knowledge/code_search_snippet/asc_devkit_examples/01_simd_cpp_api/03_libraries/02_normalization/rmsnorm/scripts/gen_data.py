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


def generate_random_data(min_range, max_range, shape, dtype):
    if isinstance(min_range, str):
        min_range = hex_str_to_fp32(min_range)
        max_range = hex_str_to_fp32(max_range)
    full_value = None
    if min_range in [np.inf, -np.inf] or np.isnan(min_range):
        full_value = min_range
    elif max_range in [np.inf, -np.inf] or np.isnan(max_range):
        full_value = max_range
    if full_value is not None:
        return np.full(shape, full_value, dtype=dtype)
    if min_range == max_range:
        return np.full(shape, min_range, dtype=dtype)
    return np.random.uniform(min_range, max_range, shape).astype(dtype)


def rms(x, epsilon=1e-8, axis=-1):
    def rms1(v, epsilon, axis=None):
        return np.sqrt(np.mean(np.square(v), axis, keepdims=True) + epsilon)
    return rms1(x, epsilon, axis)


def rms_norm(x, gamma, epsilon=1e-8, axis=-1):
    res = rms(x, epsilon)
    tmp = 1 / res
    gamma = gamma.reshape(1, 1, gamma.shape[-1])
    return x / res * gamma


def gen_golden_data_simple():
    dtype = np.float32
    src_gm = generate_random_data(-1, 1, [1, 1, 16], dtype)
    gamma_gm = generate_random_data(-1, 1, [16], dtype)
    os.makedirs("input", exist_ok=True)
    src_gm.reshape(-1).tofile("./input/input_inputX.bin")
    gamma_gm.reshape(-1).tofile("./input/input_gamma.bin")
    src_gm = src_gm.astype(np.float32)
    gamma_gm = gamma_gm.astype(np.float32)
    epsilon = np.float32(5e-05).astype(dtype)
    golden_gm = rms_norm(src_gm, gamma_gm, epsilon).astype(dtype)
    os.makedirs("output", exist_ok=True)
    golden_gm.reshape(-1).tofile("./output/golden.bin")
    tiling = np.array([1, 1, 16], dtype="uint32")
    tiling.tofile("./input/input_tiling.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
