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
    input_type = np.dtype("float16")
    output_type = input_type
    block_length = 2048
    # 一次迭代能处理256B数据
    one_repeat_items = (256 // input_type.itemsize) 
    repeat = block_length // one_repeat_items
    
    input_shape = [block_length]
    output_shape = [repeat] 
    input_x = np.ones(input_shape).astype(input_type)
    golden = np.zeros(output_shape).astype(output_type)
    for i in range(repeat):
        golden[i] = np.sum(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
