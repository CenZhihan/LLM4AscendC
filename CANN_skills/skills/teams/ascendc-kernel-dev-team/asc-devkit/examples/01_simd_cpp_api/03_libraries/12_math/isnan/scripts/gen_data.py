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
import random
import numpy as np


def random_set_boundary(src_seq):
    for seq in src_seq:
        for i in random.sample(range(len(seq)), len(seq) // 10):
            seq[i] = np.inf
    for seq in src_seq:
        for i in random.sample(range(len(seq)), len(seq) // 10):
            seq[i] = -np.inf
    for seq in src_seq:
        for i in random.sample(range(len(seq)), len(seq) // 10):
            seq[i] = -np.nan
    for seq in src_seq:
        for i in random.sample(range(len(seq)), len(seq) // 10):
            seq[i] = np.nan
    return

def gen_golden_data_simple():
    dtype = np.float32
    count = int(1024)
    data_size = int(1024)

    np.set_printoptions(threshold=sys.maxsize)

    src = np.random.uniform(1024, -1024, [1024]).astype(dtype)
    random_set_boundary([src])
    golden = np.zeros([1024]).astype(dtype)
    golden[:count] = np.isnan(src[:count]).astype(dtype)

    os.makedirs("input", exist_ok=True)
    src.tofile("./input/input_src.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
