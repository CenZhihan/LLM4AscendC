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
import torch
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
    count = int(128)
    data_size = int(128)
    np.set_printoptions(threshold=sys.maxsize)

    if dtype == np.half:
        src0 = np.random.uniform(-300, 300, [128]).astype(dtype)
        src1 = np.random.uniform(-300, 300, [128]).astype(dtype)
        src2 = np.random.uniform(-300, 300, [128]).astype(dtype)
    else:
        src0 = np.random.uniform(-pow(2, 64), pow(2, 64), [128]).astype(dtype)
        src1 = np.random.uniform(-pow(2, 64), pow(2, 64), [128]).astype(dtype)
        src2 = np.random.uniform(-pow(2, 64), pow(2, 64), [128]).astype(dtype)
    random_set_boundary([src0])
    random_set_boundary([src1])
    random_set_boundary([src2])

    golden = np.zeros([128]).astype(dtype)
    golden = torch.addcmul(torch.from_numpy(src2), torch.from_numpy(src0), torch.from_numpy(src1), value=1).numpy()
    golden[count:data_size] = 0

    os.makedirs("input", exist_ok=True)
    src0.tofile("./input/input_src0.bin")
    src1.tofile("./input/input_src1.bin")
    src2.tofile("./input/input_src2.bin")
    os.makedirs("output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
