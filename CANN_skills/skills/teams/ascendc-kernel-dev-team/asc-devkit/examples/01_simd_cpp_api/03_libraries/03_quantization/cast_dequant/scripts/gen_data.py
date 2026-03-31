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


def gen_golden_data():
    data_size = 256
    half_block = False

    input_gm = np.random.randint(0, 100, [data_size]).astype(np.int16)
    golden = np.zeros([data_size * 2], dtype=np.uint8)
    if half_block:
        for i in range(data_size * 2 // 32):
            golden[i * 32 + 16:((i * 32) + 32)] = input_gm[i * 16:((i * 16) + 16)]
    else:
        for i in range(data_size):
            golden[i * 32:((i * 32) + 16)] = input_gm[i * 16:((i * 16) + 16)]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    input_gm.tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data()
