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



def get_range_by_dtype(input_type):
    try:
        if input_type == np.float16 or input_type == np.float32 or input_type == np.float64:
            return np.finfo(input_type).min, np.finfo(input_type).max
        else:
            return np.iinfo(input_type).min, np.iinfo(input_type).max
    except ValueError:
        print(f"Unsupported data type:{input_type}")

# 取值为VSEL_CMPMASK_SPR。根据selMask在两个tensor中选取元素。selMask中有效数据的个数存在限制，
# 具体取决于源操作数的数据类型。在每一轮迭代中，根据selMask的有效位数据进行选择操作，
# 每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。
def gen_golden_data_simple():
    input_type = np.float32
    output_type = input_type
    sel_type = np.uint8
    block_length = 256

    min_val, max_val = get_range_by_dtype(input_type)
    input_shape = [block_length]
    output_shape = [block_length]
    sel_shape = [128]
    input_x = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    input_y = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    min_val, max_val = get_range_by_dtype(sel_type)
    input_sel = np.random.uniform(min_val, max_val, sel_shape).astype(sel_type)
    # 将mask转为二进制
    bin_list = [f"{num:08b}" for num in input_sel] # sel_mask是uint8类型
    bin_sel = ""
    for bin_str in bin_list:
        bin_sel += "".join(reversed(bin_str)) # 低位bit在前，所以要翻转字符串
    golden = np.ones(output_shape).astype(output_type)
    mask_num = 64
    for i in range(len(input_x)):
        if bin_sel[i % mask_num] == '0':
            golden[i] = input_y[i]
        else:
            golden[i] = input_x[i]
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    input_sel.tofile("./input/input_sel.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
