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

def gen_golden_data_simple():
    """
    生成PairReduceSum样例的测试数据和golden数据
    输入：128个float16类型的随机数
    输出：64个float16类型，每两个相邻元素相加
    """
    input_type = np.float16
    output_type = input_type
    block_length = 128
    
    min_val, max_val = -1000, 1000
    input_shape = [block_length]
    output_shape = [block_length // 2] 
    input_x = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    golden = np.zeros(output_shape).astype(output_type)
    for i in range(len(golden)):
        golden[i] = input_x[i * 2] + input_x[i * 2 + 1]
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
