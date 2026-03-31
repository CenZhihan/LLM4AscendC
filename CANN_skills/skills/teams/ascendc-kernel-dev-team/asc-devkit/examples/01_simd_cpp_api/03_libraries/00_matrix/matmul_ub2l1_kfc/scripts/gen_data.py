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


def get_tiling():
    m = 32
    n = 32
    k = 32
    tiling = np.zeros((32), dtype=np.int32)
    used_core_num = 1
    single_core_m = 32
    single_core_n = 32
    single_core_k = 32
    base_m = 32
    base_n = 32
    base_k = 32
    depth_a1 = 1
    depth_b1 = 1
    step_m = 1
    step_n = 1
    is_bias = 1
    trans_length = 4096
    iterate_order = 0
    shared_mode = 0
    l1_size = 4224
    l0c_size = 4096
    ub_size = 8192
    step_ka = 1
    step_kb = 1
    tiling[0:32] = [used_core_num, m, n, k, k, single_core_m, single_core_n, single_core_k, base_m, base_n, base_k,
                    depth_a1, depth_b1, step_m, step_n, is_bias, trans_length, iterate_order, 0, l1_size, l0c_size,
                    ub_size, 0, 0, 0, 0, step_ka, step_kb, 0, 0, 0, 0]
    return tiling


def gen_golden_data_simple():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    m = 32
    n = 32
    k = 32
    x1_gm = np.ones(m * k).reshape([m, k]).astype(np.float16)
    x2_gm = np.ones(k * n).reshape([k, n]).astype(np.float16)
    golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    tiling = get_tiling()
    workspace = np.zeros(131072).astype(np.float32)
    x1_gm.tofile("./input/input_x.bin")
    x2_gm.tofile("./input/input_y.bin")
    tiling.tofile("./input/tiling.bin")
    workspace.tofile("./input/workspace.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
