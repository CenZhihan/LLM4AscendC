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

def gemm_custom_tiled(m, k, n, A, B, C, k0):
    """
    执行分块矩阵乘加运算: C = A * B^T + C

    参数：
    m, k, n : 矩阵的维度参数(显式传入)
    A       :形状为(m, k)的矩阵
    B       :形状为(n, k)的矩阵
    C       :形状为(m, n)的矩阵(包含初始值)
    k0      :内积方向上的分块大小(Block size)
    """

    # 0. 维度校验(虽然传入了m, k, n， 但在Python中最好校验一下实际矩阵是否符合)
    if A.shape != (m, k) or B.shape != (n, k) or C.shape != (m, n):
        raise ValueError(f"矩阵实际尺寸与传入的m = {m}, k = {k}, n = {n} 不匹配")
    
    print(f"正在计算: C({m}x{n}) += A({m}x{k}) * B.T({k}x{n}) | 分块 k0={k0}")

    # 1. 外层循环： 遍历行(m)
    for i in range(m):
        # 2. 中层循环：遍历列(n)
        for j in range(n):
            # 【初始化累加器】：直接读取 C 矩阵当前位置的值
            # 相当于公式中的 "+C" 部分
            accumulate_sum = C[i, j]

            # 3. 内层循环：在k维度上分块前进
            # range(start, stop, step) -> 每次跳跃 k0
            for p in range(0, k, k0):

                # 计算当前块的终点，处理剩余长度不足 k0 的边界情况
                p_end = min(p + k0, k)

                # 获取 A 的行片段: A[i, p ~ p_end]
                vec_a_chunk = A[i, p:p_end]

                # 获取 B 的行片段: B[j, p ~ p_end]
                # 注意：因为 B 是(n, k)，计算内积时直接取 B 的第 j 行对应片段即可
                vec_b_chunk = B[j, p:p_end]

                # 【核心计算】：计算这k0个数字的内积，并加到累加器上
                accumulate_sum += np.dot(vec_a_chunk, vec_b_chunk)

                print(p)

            # 4. 写回结果：将累加完成的值更新回 C 矩阵
            C[i, j] = accumulate_sum

    return C

# ---测试代码---
if __name__ == "__main__":
    # 1. 定义维度参数
    m_in = 100
    k_in = 200
    n_in = 50
    block_size = 16

    # 2. 准备数据
    np.random.seed(42)
    A_mat = np.random.rand(m_in, k_in)
    B_mat = np.random.rand(n_in, k_in)  # 注意形状是 (n, k)
    C_mat = np.random.rand(m_in, n_in)  # C 包含初始噪音/偏置值

    # 备份一份 C 用于后续验证，因为上面的函数会直接修改 C_mat
    C_original = C_mat.copy()

    # 3. 调用函数(m, k, n 作为入参)
    # Python 传递的是引用， C_mat 会被直接修改 (In-place update)
    gemm_custom_tiled(m_in, k_in, n_in, A_mat, B_mat, C_mat, block_size)

    # 4. 验证结果
    # 标准计算公式: C_old + A @ B.T
    C_standard = C_original + np.dot(A_mat, B_mat.T)

    if np.allclose(C_mat, C_standard):
        print("验证通过: 结果与 Numpy 标准计算一致")
        print(f"   C[0, 0]最终值: {C_mat[0, 0]:.4f}")
    else:
        print("验证失败")