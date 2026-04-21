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


def softmax_grad_float(grad, src, isFront = None):
    muls_r = grad * src
    muls_r = muls_r.sum(axis=-1, keepdims=True)
    if isFront :
        return muls_r
    sub_r = grad - muls_r
    res = sub_r * src
    return res

def gen_golden_data_simple():
    x_shape = (960, 960)
    workspace_shape = (1024)
    golden_shape = (960, 8)
    x = np.random.uniform(-1, 1, x_shape).astype(np.float32)
    y = np.random.uniform(-1, 1, x_shape).astype(np.float32)
    z = np.random.uniform(0, 0, golden_shape).astype(np.float32)
    workspace = np.random.uniform(0, 0, workspace_shape).astype(np.uint32)

    golden = softmax_grad_float(x, y, True)
    golden = golden + z
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")
    workspace.tofile("./input/workspace.bin")

if __name__ == "__main__":
    gen_golden_data_simple()