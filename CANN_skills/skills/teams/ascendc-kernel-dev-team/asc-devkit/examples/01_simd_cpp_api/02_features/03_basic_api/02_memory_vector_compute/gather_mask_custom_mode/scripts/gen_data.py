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
from enum import Enum



def get_range_by_dtype(input_type):
    try:
        if input_type == np.float16 or input_type == np.float32 or input_type == np.float64:
            return np.finfo(input_type).min, np.finfo(input_type).max
        else:
            return np.iinfo(input_type).min, np.iinfo(input_type).max
    except ValueError:
        print(f"Unsupported data type:{input_type}")


def gen_golden_data_simple():
    one_repeat_size = 256 #一次迭代能处理256B的数据
    data_block_size = 32 #一个datablock大小是32B
    input_type = np.dtype("uint32")
    output_type = input_type
    type_size = input_type.itemsize 
    block_length = 256
    mask = 70
    src0_block_stride = 1 
    repeat_times = 2      
    src0_repeat_stride = 4 
    src1_repeat_stride = 0 
    one_data_block_items = data_block_size // type_size

    min_val, max_val = get_range_by_dtype(input_type)
    input_x_shape = [block_length]
    input_y_shape = [block_length // 8] #mask一个bit对应一个数
    input_x = np.random.uniform(min_val, max_val, input_x_shape).astype(input_type) 
    input_y = 0x7E7C00A5 * np.ones(input_y_shape).astype(input_type)
    input_mask = np.unpackbits(input_y.view(np.uint8), bitorder='little').astype(bool)
    golden = np.zeros(input_x_shape).astype(input_type)
    for i in range(repeat_times):
        base = i * (src0_repeat_stride // src0_block_stride) * one_data_block_items
        base_m = i * src1_repeat_stride * one_data_block_items
        src0_iter = input_x[base : base + mask]
        selected = src0_iter[input_mask[base_m : base_m + mask]]
        base_g = i * selected.size
        golden[base_g : base_g + selected.size] = selected
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")
    
if __name__ == "__main__":
    gen_golden_data_simple()
