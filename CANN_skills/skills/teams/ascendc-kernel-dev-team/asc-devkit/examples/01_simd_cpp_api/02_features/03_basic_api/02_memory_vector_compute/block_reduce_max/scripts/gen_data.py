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
    """
    根据数据类型获取取值范围
    Args:
        input_type: numpy数据类型
    Returns:
        tuple: (最小值, 最大值)
    """
    try:
        if input_type == np.float16 or input_type == np.float32 or input_type == np.float64:
            return np.finfo(input_type).min, np.finfo(input_type).max
        else:
            return np.iinfo(input_type).min, np.iinfo(input_type).max
    except ValueError:
        print(f"Unsupported data type:{input_type}")

# 返回结果中索引index数据是按照dst的数据类型进行存储的，比如dst使用half类型时，
# index按照half类型进行存储，如果按照half格式进行读取，index的值是不对的，因此index的读取需要使用reinterpret_cast方法转换到整数类型。
# 若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用的是reinterpret_cast<uint32_t*>
def gen_golden_data_simple():
    """
    生成BlockReduceMax样例的测试数据和golden数据
    输入：128个float16类型的随机数
    输出：64个float16类型，每个datablock内所有元素的最大值
    """
    input_type = np.dtype("float16")
    output_type = input_type
    max_index_type = np.uint16 #与half对应
    block_length = 128
    #一个datablock处理32B数据
    one_data_block_items = (32 // input_type.itemsize) 
    block_num = block_length // one_data_block_items

    min_val, max_val = get_range_by_dtype(input_type)
    input_shape = [block_length]
    output_shape = [64] 
    input_x = np.random.uniform(min_val, max_val, input_shape).astype(input_type)
    golden = np.zeros(output_shape).astype(output_type)
    for i in range(block_num):
        #获取最大值
        golden[i] = np.max(input_x[i * one_data_block_items:(i + 1) * one_data_block_items])
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
