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


import numpy as np
import os


def fix_idx_mat(idx_mat):
    for i in range(idx_mat.shape[1]):
        for j in range(idx_mat.shape[0]):
            if j % 2 == 0:
                if (idx_mat[j][i] - idx_mat[j+1][i]) == 1:
                    idx_mat[j][i] -= 1
                elif (idx_mat[j][i] - idx_mat[j+1][i]) == 2:
                    idx_mat[j][i] -= 2


def denst_idx(idx_mat):
    idx_mat_golden = np.zeros(\
        shape=(idx_mat.shape[0], idx_mat.shape[1],\
               idx_mat.shape[2], int(idx_mat.shape[3] / 4)), dtype=np.uint8)
    for i in range(idx_mat.shape[0]):
        for j in range(idx_mat.shape[1]):
            for k in range(idx_mat.shape[2]):
                for z in range(idx_mat.shape[3]):
                    if z % 4 == 0:
                        idx_mat_golden[i][j][k][z // 4] = idx_mat[i][j][k][z] | \
                        (idx_mat[i][j][k][z + 1] << 2) | (idx_mat[i][j][k][z + 2] << 4) | \
                        (idx_mat[i][j][k][z + 3] << 6)
    return idx_mat_golden


def dense_tensor(tensor_a, tensor_idx):
    result = np.zeros(shape=(int(tensor_a.shape[0]/2)), dtype=np.uint8)
    for i in range(result.shape[0]):
        if i % 2 == 0:
            mat_idx = i * 2
            result[i] = tensor_a[mat_idx + tensor_idx[i]]
        else:
            result[i] = tensor_a[mat_idx + tensor_idx[i] + 1]
    return result


def cal_sparse_matmul(mat_a, mat_b, mat_idx):
    result = np.zeros(shape=(mat_a.shape[0], mat_b.shape[1]), dtype=np.int32)
    for i in range(mat_idx.shape[1]):
        for j in range(mat_a.shape[0]):
            densed_a = dense_tensor(mat_a[j, :], mat_idx[:, i])
            result[j][i] = np.sum(densed_a * mat_b[:, i]).astype(np.int32)
    return result


def gen_golden_data():
    M = 16
    N = 16
    K = 64

    x1_gm = np.random.uniform(1, 10, [16, 64]).astype(np.int8)
    x2_gm = np.random.uniform(1, 10, [32, 16]).astype(np.int8)
    idx_gm = np.random.randint(0, 3, (32, 16)).astype(np.uint8)
    fix_idx_mat(idx_gm)
    golden = cal_sparse_matmul(x1_gm, x2_gm, idx_gm)
    c0Size = 32
    x1_gm = x1_gm.reshape((int(M / 16), 16, int(K / c0Size), c0Size))\
        .transpose(0, 2, 1, 3).astype(np.int8)
    x2_gm = x2_gm.reshape((int(K / 2 / c0Size), c0Size, int(N / 16), 16))\
        .transpose(0, 2, 3, 1).astype(np.int8)
    idx_gm = idx_gm.reshape((int(K / 2 / c0Size), c0Size, int(N / 16), 16))\
        .transpose(0, 2, 3, 1).astype(np.uint8)
    idx_gm_golden = denst_idx(idx_gm)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    idx_gm_golden.tofile("./input/idx_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
