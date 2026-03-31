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


import numpy as np
import os

def gen_golden_data_simple():
    input_x_ = np.arange(11*16*4).astype(np.float16)*(-1)
    zero_tensor = np.zeros([16-11]).astype(np.float16)
    input_x = np.concatenate((input_x_, zero_tensor),axis=None)
    golden = np.absolute(input_x).astype(np.float16)
    sync_tensor = np.zeros([8*4]).astype(np.int32)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")
    sync_tensor.tofile("./input/sync.bin")

if __name__ == '__main__':
    gen_golden_data_simple()