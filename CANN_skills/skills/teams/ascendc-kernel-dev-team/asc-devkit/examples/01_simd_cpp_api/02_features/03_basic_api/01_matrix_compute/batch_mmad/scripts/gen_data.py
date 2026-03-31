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
import argparse
import numpy as np


def gen_golden_data(B=3,M=0,K=0,N=0):
    #带batch的矩阵乘，输入[B, M, K]和[B, K, N]
    input_type = np.dtype("float32")
    golden = np.zeros([B, M, N]).astype(np.float32)
    x1_gm = np.random.uniform(1, 10, [B, M, K]).astype(input_type)
    x2_gm = np.random.uniform(1, 10, [B, K, N]).astype(input_type)
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))
    golden = golden.astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=1, choices=range(1,13))
    parser.add_argument('-m', type=int)
    parser.add_argument('-k', type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    gen_golden_data(args.b, args.m, args.k, args.n)
