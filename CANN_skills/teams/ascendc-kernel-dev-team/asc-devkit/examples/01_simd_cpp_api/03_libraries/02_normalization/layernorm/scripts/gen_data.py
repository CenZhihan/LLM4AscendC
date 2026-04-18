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
    shape = [2, 32, 16]
    inputX = np.arange(shape[0]*shape[1]*shape[2]).reshape(shape).astype(np.float32)
    gamma = np.arange(shape[2]).reshape(shape[2]).astype(np.float32)
    beta = np.arange(shape[2]).reshape(shape[2]).astype(np.float32)

    reduce_axis = 2
    mean  = np.mean(inputX, reduce_axis, keepdims=True)
    variance = np.mean(np.power((inputX - mean),2), reduce_axis, keepdims=True)
    result = gamma*((inputX - mean) / np.sqrt(variance + 0.0001)) + beta

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    inputX.tofile("./input/input_inputX.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    result.tofile("./output/golden_output_result.bin")
    mean.tofile("./output/golden_output_mean.bin")
    variance.tofile("./output/golden_output_variance.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
