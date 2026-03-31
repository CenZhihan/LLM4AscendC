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


def softmax_py_float(x):
    """
    Compute the softmax function for each channel of the input x.
    """
    orig_shape = x.shape
    x_max = np.max(x, axis=-1)
    x_max = np.reshape(x_max, [orig_shape[0], 1])

    x_sub = x - x_max
    x_exp = np.exp(x_sub)

    x_exp1 = np.reshape(x_exp, [orig_shape[0], orig_shape[1]])
    x_sum = np.sum(x_exp1, axis=-1)
    x_sum = np.reshape(x_sum, [orig_shape[0], 1])
    x_div = x_exp / x_sum
    out = np.reshape(x_div, [orig_shape[0], orig_shape[1]])
    return out, x_max, x_sum

def gen_golden_data_simple():
    x_shape = (960, 960)
    workspace_shape = (1024)
    x = np.random.uniform(-1, 1, x_shape).astype(np.float32)
    input_max = np.zeros([x_shape[0], 8], dtype=np.float32)
    input_sum = np.zeros([x_shape[0], 8], dtype=np.float32)
    workspace = np.random.uniform(0, 0, workspace_shape).astype(np.uint32)

    golden, max, sum = softmax_py_float(x)
    input_max = input_max + max
    input_sum = input_sum + sum

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile("./input/input_x.bin")
    input_max.tofile("./input/input_max.bin")
    input_sum.tofile("./input/input_sum.bin")
    workspace.tofile("./input/workspace.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
