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


def adjust_softmax_res(res, max_val, res_shape):
    target = 0xFF7FFFFF
    to = 0.0
    for i in range(res_shape[0]):
        if max_val[i][0] == target:
            for j in range(res_shape[1]):
                res[i][j] = to
    return


def softmax_py_float(x):
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
    x_shape = (32, 32)
    x = np.random.uniform(-1, 1, x_shape).astype(np.float32)

    output_max = np.zeros([x_shape[0], 8], dtype=np.float32)

    softmax_out, max_val, sum_val = softmax_py_float(x)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    softmax_out.tofile("./input/input_softmax.bin")

    adjust_softmax_res(softmax_out, max_val, softmax_out.shape)
    output_max = output_max + max_val

    output_max.tofile("./input/input_max.bin")
    softmax_out.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()