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
    B = 2
    S = 32
    H = 16
    eps = 0.0001

    shapeDy = [B, S, H]
    shapeX = [B, S, H]
    shapeGamma = [H]

    inputDy = np.random.uniform(-100, 100, shapeDy).astype(np.float32)
    inputX = np.random.uniform(-100, 100, shapeX).astype(np.float32)
    inputGamma = np.random.uniform(-100, 100, shapeGamma).astype(np.float32)

    reduce_axis = 2

    inputMean = np.mean(inputX, axis=reduce_axis, keepdims=True)
    inputVariance = np.mean(
        np.power((inputX - inputMean), 2), axis=reduce_axis, keepdims=True)
    
    os.makedirs("input", exist_ok=True)
    inputDy.tofile("./input/input_inputDy.bin")
    inputX.tofile("./input/input_inputX.bin")
    inputMean.tofile("./input/input_inputMean.bin")
    inputVariance.tofile("./input/input_inputVariance.bin")
    inputGamma.tofile("./input/input_inputGamma.bin")

    pd_xl = np.multiply(inputDy, inputGamma)
    inputMean_brc = np.broadcast_to(inputMean, shapeDy)
    tmp1Tensor = np.subtract(inputX, inputMean_brc)
    tmp2Tensor = np.power(np.add(inputVariance, eps), -0.5)
    reciprocal = np.divide(1.0, shapeDy[-1])
    pd_var = np.sum((np.multiply(np.multiply(np.multiply(-0.5, pd_xl), tmp1Tensor),
                    np.power(np.add(inputVariance, eps), -1.5))), reduce_axis, keepdims=True)
    pd_mean = np.add(
        np.sum(np.multiply(np.multiply(-1.0, pd_xl), tmp2Tensor),
               reduce_axis, keepdims=True),
        np.multiply(np.multiply(pd_var, reciprocal), np.sum(np.multiply(-2.0, tmp1Tensor),
                                                            reduce_axis, keepdims=True)))
    pd_x = np.add(np.add(np.multiply(pd_mean, reciprocal), np.multiply(np.multiply(
        pd_var, np.divide(2.0, shapeDy[-1])), tmp1Tensor)), np.multiply(pd_xl, tmp2Tensor))
    res_for_gamma = np.multiply(tmp1Tensor, tmp2Tensor)

    os.makedirs("output", exist_ok=True)
    pd_x.tofile("./output/golden_outputPdX.bin")
    res_for_gamma.tofile("./output/golden_resForGamma.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
