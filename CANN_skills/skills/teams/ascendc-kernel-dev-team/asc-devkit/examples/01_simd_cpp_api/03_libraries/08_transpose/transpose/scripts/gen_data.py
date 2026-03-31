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
    dtype = np.float16

    b, n, s, hn_div = 1, 2, 64, 32
    
    origin_shape = (b, n, s, hn_div)

    origin_data = np.random.uniform(-1000.0, 1000.0, origin_shape).astype(dtype)

    reshape_shape = (b, n, s // 16, 16, hn_div // 16, 16)
    reshaped_data = origin_data.reshape(reshape_shape)

    transposed_data = np.transpose(reshaped_data, (0, 1, 4, 2, 3, 5))

    golden = np.transpose(origin_data, (0, 2, 1, 3))

    os.system("mkdir -p ./input")
    transposed_data.tofile("./input/input_x.bin")
    os.system("mkdir -p ./output")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
