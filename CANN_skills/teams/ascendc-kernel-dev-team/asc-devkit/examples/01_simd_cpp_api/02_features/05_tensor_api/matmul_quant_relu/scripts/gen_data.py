#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import numpy as np


def saturation(value, min_val, max_val, target_type):
    """
    将输入的浮点数进行饱和处理，并转换为目标类型
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)


def gen_golden_data():
    input_shape_x = [128, 96]
    input_shape_y = [96, 128]
    dtype = np.half
    input_x  = np.random.uniform(-2, 2, input_shape_x).astype(dtype)
    input_y  = np.random.uniform(-2, 2, input_shape_y).astype(dtype)
    golden = np.matmul(input_x.astype(np.float32), input_y.astype(np.float32)).astype(np.float32)

    # relu
    golden = np.maximum(0, golden)

    # vector quant
    temp_quant_tensor = np.random.uniform(low=-2, high=2, size=128).astype(np.float32)
    uint32_deq_scale = np.frombuffer(temp_quant_tensor, np.uint32)
    uint32_deq_scale &= 0XFFFFE000
    quant_tensor = uint32_deq_scale.astype(np.uint64)
    quant_tensor |= 1 << 46
    golden = golden * temp_quant_tensor
    golden = saturation(golden, -128, 127, np.int8)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    quant_tensor.tofile("./input/quant.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
    