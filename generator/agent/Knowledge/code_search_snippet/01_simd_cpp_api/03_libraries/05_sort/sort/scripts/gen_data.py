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
    x1 = np.random.uniform(-60000, 60000, [128]).astype(np.float32)
    x2 = np.zeros([128]).astype(np.int32)
    golden = (-np.sort(-x1)).astype(np.float32)
    
    os.makedirs("input", exist_ok=True)
    x1.tofile("./input/input_srcGmValue.bin")
    x2.tofile("./input/input_srcGmIndex.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden_dstGmValue.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
