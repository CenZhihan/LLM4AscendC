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
    zero_tensor = np.zeros([16 - 4]).astype(np.float16)
    sync_tensor = np.zeros([8 * 4]).astype(np.int32)
    input_ = np.arange(4 * 4 * 4).astype(np.float16) * (-1)
    input_x = np.concatenate((input_, zero_tensor), axis=None)
    zero_tensor = np.zeros([16 * 4]).astype(np.float16)
    for i in range(0, 61):
        if i % 4 == 0:
            zero_tensor[i] = i * (-1) - 3
    input_x = zero_tensor
    zero_gm = np.zeros([16 - 4]).astype(np.float16)

    golden = np.concatenate((input_x, zero_gm), axis=None)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")
    sync_tensor.tofile("./input/sync.bin")

if __name__ == '__main__':
    gen_golden_data_simple()