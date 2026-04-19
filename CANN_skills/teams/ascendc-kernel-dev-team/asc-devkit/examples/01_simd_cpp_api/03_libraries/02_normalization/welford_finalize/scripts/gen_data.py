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

    counts_dtype = np.int32
    input_mean_shape = [1, 1024]
    input_mean_dtype = np.float32
    input_variance_shape = [1, 1024]
    input_variance_dtype = np.float32
    output_mean_shape = [8]
    output_mean_dtype = np.float32
    output_variance_shape = [8]
    output_variance_dtype = np.float32

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=False)
    np.random.seed(19)
    r_length = 4096
    head = 4
    head_length = 1024
    tail = 4
    tail_length = 0
    is_counts = 1
    ab_length = 1024
    ab_recf = 1.0 / ab_length
    n_recf = 1.0 / r_length
    rn_length = 1
    counts = np.random.randint(0, 1, [1024]).astype(counts_dtype)

    for i in range(tail_length):
        counts[head_length + i] = tail

    for i in range(head_length):
        counts[i] = head

    inputmean = np.random.uniform(-2, 2, input_mean_shape).astype(input_mean_dtype)
    inputvar = np.random.uniform(-3, 3, input_variance_shape).astype(input_variance_dtype)
    n_rec = np.random.uniform(n_recf, n_recf, input_mean_shape).astype(np.float32)
    ab_rec = np.random.uniform(ab_recf, ab_recf, input_mean_shape).astype(np.float32)

    os.makedirs("input", exist_ok=True)
    counts.tofile("./input/input_counts.bin")
    inputmean.tofile("./input/input_inMean.bin")
    inputvar.tofile("./input/input_inVar.bin")

    newcounts = np.random.randint(0, 1, input_mean_shape).astype(counts_dtype)
    for i in range(input_mean_shape[0]):
        for j in range(input_mean_shape[1]):
            newcounts[i][j] = counts[j]

    if counts_dtype != np.float32:
        counts = counts.astype(np.float32)
        inputmean = inputmean.astype(np.float32)
        inputvar = inputvar.astype(np.float32)

    if is_counts == 1 or tail_length != 0:
        mean = np.sum(inputmean * newcounts, axis=-1)
        mean = np.reshape(mean, [input_mean_shape[0], 1])
        mean = mean * n_rec
        variance = np.sum(inputvar + counts * np.power((inputmean - mean), 2), axis=-1)
        variance = np.reshape(variance, [input_mean_shape[0], 1])
        variance = variance * n_rec
    else: 
        mean = np.sum(inputmean, axis=-1)
        mean = np.reshape(mean, [input_mean_shape[0], 1])
        mean = mean * ab_rec
        variance = np.sum(inputvar + rn_length * np.power((inputmean - mean), 2), axis=-1)
        variance = np.reshape(variance, [input_mean_shape[0], 1])
        variance = variance * n_rec

    outmean = np.random.uniform(0, 0, output_mean_shape).astype(output_mean_dtype)
    outvar = np.random.uniform(0, 0, output_variance_shape).astype(output_variance_dtype)
    for i in range(input_mean_shape[0]):
        outmean[i] = mean[i][0]
        outvar[i] = variance[i][0]

    os.makedirs("output", exist_ok=True)
    outmean.tofile("./output/golden_outMean.bin")
    outvar.tofile("./output/golden_outVar.bin")

if __name__ == "__main__":
    gen_golden_data_simple()