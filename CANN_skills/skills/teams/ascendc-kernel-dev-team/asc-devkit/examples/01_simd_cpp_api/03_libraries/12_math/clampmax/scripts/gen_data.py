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
import math
import numpy as np


def gen_golden_data_simple():
    # 初始化参数
    dtype = np.float32
    src_shape = [256]
    data_size = 256
    min_value = 65500
    min_num = (np.array([65500]).astype(dtype))[0]
    max_num = (np.array([65504]).astype(dtype))[0]
    scalar = (np.array([65502]).astype(dtype))[0]
    # 产生随机的输入数据
    src = np.random.uniform(min_num, max_num, src_shape).astype(dtype)
    if min_value == -65504:
        for i in range(data_size):
            if i % 9 == 0:
                src[i] = -60000 - 65500
    src_max = np.max(src).astype(dtype)
    src_min = np.min(src).astype(dtype)
    # 生成 golden 数据
    n = data_size - 0
    golden = src[:n]
    clamp_type = 1
    if clamp_type == 0:
        golden = np.clip(src, a_min=scalar, a_max=None)
    else:
        golden = np.clip(src, src_min, scalar)
    # 保存数据文件
    os.makedirs("input", exist_ok=True)
    src.tofile("./input/input_x.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
