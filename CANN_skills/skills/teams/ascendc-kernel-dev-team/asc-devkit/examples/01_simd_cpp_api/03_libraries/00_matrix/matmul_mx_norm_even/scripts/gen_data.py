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
import ml_dtypes

def gen_golden_data():
    m = 32
    n = 128
    k = 128
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    is_bias = False
    is_trans_a = False
    is_trans_b = False
    is_trans_scalea = False
    is_trans_scaleb = False

    src_type = ml_dtypes.float8_e5m2
    src_scale_type = np.uint8
    dst_type = np.float32
    c0size = 16
    sk = (int)(np.ceil(k / 64) * 2)

    if is_trans_a:
        x1_shape = [k, m]
    else:
        x1_shape = [m, k]
    
    if is_trans_b:
        x2_shape = [n, k]
    else:
        x2_shape = [k, n]
    
    x1_s_shape = [m, sk]
    x2_s_shape = [sk, n]

    x1_gm = (np.random.randint(-30, 30, x1_shape) * 0.1).astype(src_type)
    x2_gm = (np.random.randint(-30, 30, x2_shape) * 0.1).astype(src_type)
    bias_gm = np.random.randint(0, 10, (1, n)).astype(dst_type)
    x1_mx_gm = np.random.randint(127, 130, x1_s_shape).astype(src_scale_type)
    x2_mx_gm = np.random.randint(127, 130, x2_s_shape).astype(src_scale_type)

    x1_mx = 2 ** (x1_mx_gm.astype(np.float64) - 127)
    x2_mx = 2 ** (x2_mx_gm.astype(np.float64) - 127)
    
    x1 = np.zeros(x1_shape, dtype=np.float64)
    x2 = np.zeros(x2_shape, dtype=np.float64)

    for i in range(x1_gm.shape[1]):
        x1[:, i] = x1_gm[:, i] * x1_mx[:, i // 32]
        x2[i, :] = x2_gm[i, :] * x2_mx[i // 32, :]
    
    y_gm = np.matmul(x1.astype(np.float64), x2.astype(np.float64)).astype(dst_type)

    if is_trans_scalea:
        x1_mx_gm = x1_mx_gm.transpose()
    if is_trans_scaleb:
        x2_mx_gm = x2_mx_gm.transpose()
    
    if is_trans_scalea:
        x1_mx_gm = x1_mx_gm.reshape(int(sk / 2), 2, m).transpose(0, 2, 1)
    if not is_trans_scaleb:
        x2_mx_gm = x2_mx_gm.reshape(int(sk / 2), 2, n).transpose(0, 2, 1)

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    x1_mx_gm.tofile("./input/x1_mx_gm.bin")
    x2_mx_gm.tofile("./input/x2_mx_gm.bin")

    y_gm.tofile("./output/golden.bin")
    if is_bias:
        bias_gm.tofile("/input/bias_gm.bin")


if __name__ == "__main__":
    gen_golden_data()
    