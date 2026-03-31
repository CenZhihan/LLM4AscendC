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
import sys
import numpy as np


def gen_golden_data_simple():
    dtype = np.float32
    src_shape = [8]

    np.set_printoptions(threshold=sys.maxsize)
    src = np.random.uniform(-np.pi, np.pi, src_shape).astype(dtype)
    cal_count = 8
    if (cal_count != 0):
        src[0:int(cal_count / 8)] = np.nan
        src[int(cal_count / 8):int(cal_count / 4)] = np.inf
    else:
        src[0:int(src_shape[0] / 8)] = np.nan
        src[int(src_shape[0] / 8):int(src_shape[0] / 4)] = -np.inf
    golden = np.zeros(src_shape).astype(dtype)
    for i, _ in enumerate(src):
        golden[i] = np.trunc(src[i])
    if (cal_count != 0):
        golden[cal_count:] = 0

    os.makedirs("input", exist_ok=True)
    src.tofile("./input/input_x.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
