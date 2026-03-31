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
import copy
import torch
import numpy as np


def binary_cumsum(src_gm):
    dst = copy.deepcopy(src_gm)
    if src_gm.dtype == np.float16:
        dst = dst.astype(np.float32)

    colume_origin = src_gm.shape[0]
    colume = np.power(2, np.ceil(np.log2(colume_origin)))
    round_cumsum = int(colume / 2)
    cur_round = 1
    while round_cumsum >= 1:
        cur_round2 = pow(2, cur_round)
        for index in range(0, int(colume / cur_round2)):
            line0 = pow(2, cur_round - 1) - 1 + index * cur_round2
            for j in range(1, pow(2, cur_round - 1) + 1):
                line1 = line0 + j
                if line1 > colume_origin - 1:
                    break
                dst[line1, ] = dst[line0, ] + dst[line1, ]
            round_cumsum = int(round_cumsum / 2)
            cur_round += 1
    if src_gm.dtype == np.float16:
        dst = dst.astype(np.float16)
    return dst


def linebyline_cumsum(src_gm):
    dst = copy.deepcopy(src_gm)
    if src_gm.dtype == np.float16:
        dst = dst.astype(np.float32)

    row_count, col_count = src_gm.shape
    for i in range(row_count - 1):
        row = i
        next_row = row + 1
        for j in range(col_count):
            dst[next_row][j] = dst[next_row][j] + dst[row][j]

    if src_gm.dtype == np.float16:
        dst = dst.astype(np.float16)
    return dst


def gen_golden_data_simple():
    dtype = np.float32
    is_last_axis = 0
    min_range = -10
    max_range = 10
    src_shape = [32, 160]
    algorithm = 0
    src_gm = np.random.uniform(min_range, max_range, src_shape).astype(dtype)

    if algorithm == 1:
        if is_last_axis:
            dst = binary_cumsum(src_gm.T)
            golden = dst.T
        else:
            dst = binary_cumsum(src_gm)
            golden = dst
    else:
        if is_last_axis:
            dst = linebyline_cumsum(src_gm.T)
            golden = dst.T
        else:
            dst = linebyline_cumsum(src_gm)
            golden = dst

    tiling = np.array([32, 160], dtype="uint32")
    
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    src_gm.reshape(-1).tofile("./input/input_x.bin")
    tiling.tofile("./input/input_tiling.bin")
    
    golden_last_row_gm = golden[-1, :]
    golden.reshape(-1).tofile("./output/golden_output_result.bin")
    golden_last_row_gm.tofile("./output/golden_output_last_row.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
