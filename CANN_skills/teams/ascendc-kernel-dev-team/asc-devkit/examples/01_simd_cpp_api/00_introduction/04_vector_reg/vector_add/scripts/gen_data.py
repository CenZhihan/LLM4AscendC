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
    inner = 64
    outter = 64
    data_type = np.float32
    src = np.ones([outter, inner]).astype(data_type)
    golden = np.copy(src)
    for i in range(outter - 1):
        golden[i + 1] = golden[i] + golden[i + 1]
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src.tofile('./input/input_x.bin')
    golden.tofile('./output/golden.bin')

if __name__ == "__main__":
    gen_golden_data_simple()