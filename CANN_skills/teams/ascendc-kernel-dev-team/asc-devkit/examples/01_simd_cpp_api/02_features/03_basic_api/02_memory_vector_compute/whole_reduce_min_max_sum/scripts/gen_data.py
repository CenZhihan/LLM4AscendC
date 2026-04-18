#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import os
import struct
import argparse
import numpy as np
np.random.seed(9)


def gen_golden_data(scenarioNum=1):
    """
    根据场景编号生成输入数据和Golden数据
    场景1：WholeReduceMax，输入half类型[1, 1024]，输出[1, 8]，每个repeat内求最大值（不返回索引）
    场景2：WholeReduceMin，输入half类型[1, 1024]，输出[1, 16]，每个repeat内求最小值及索引
    场景3：WholeReduceSum，输入float类型[1, 2048]，输出[1, 32]，每个repeat内求和
    场景4：RepeatReduceSum，输入float类型[1, 2048]，输出[1, 32]，累加模式求和
    场景5：WholeReduceMax + GetReduceRepeatMaxMinSpr，输入[1, 1024]，输出[1, 16]，求全局最大值及索引
    场景6：WholeReduceMin + GetReduceRepeatMaxMinSpr，输入[1, 1024]，输出[1, 16]，求全局最小值及索引
    场景7：WholeReduceSum非对齐场景，输入float类型[13, 57]，输出[1, 13]，对每行非对齐数据求和
    """
    input_type = np.dtype("float16")
    output_type = input_type
    one_repeat_items = 256 // input_type.itemsize

    if scenarioNum == 3 or scenarioNum == 4 or scenarioNum == 7:
        input_type = np.dtype("float32")
        output_type = input_type
        one_repeat_items = 256 // input_type.itemsize

    if scenarioNum == 1:
        block_length = 1024
        repeat = block_length // one_repeat_items
        input_x = np.random.uniform(-1, 1, [block_length]).astype(input_type)
        golden = np.zeros(repeat).astype(output_type)
        for i in range(repeat):
            golden[i] = np.max(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
    elif scenarioNum == 2:
        block_length = 1024
        repeat = block_length // one_repeat_items
        input_x = np.random.uniform(1, 10, [block_length]).astype(input_type)
        golden = np.zeros(2 * repeat).astype(output_type)
        for i in range(repeat):
            golden[2 * i] = np.amin(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
            min_index = np.argmin(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
            min_index = np.uint16(min_index)
            golden[2 * i + 1] = min_index.view(np.float16)
    elif scenarioNum == 3:
        block_length = 2048
        repeat = block_length // one_repeat_items
        input_x = np.random.uniform(1, 10, [block_length]).astype(input_type)
        golden = np.zeros(repeat).astype(output_type)
        for i in range(repeat):
            golden[i] = np.sum(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
    elif scenarioNum == 4:
        block_length = 2048
        repeat = block_length // one_repeat_items
        input_x = np.ones([block_length]).astype(input_type)
        golden = np.zeros(repeat).astype(output_type)
        for i in range(repeat):
            golden[i] = np.sum(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
    elif scenarioNum == 5:
        block_length = 1024
        input_x = np.random.uniform(-1000, 1000, [block_length]).astype(input_type)
        golden = np.zeros(2).astype(output_type)
        it = np.argmax(input_x)
        idx = np.uint16(it)
        golden[0] = input_x[it]
        golden[1] = idx.view(np.float16)
    elif scenarioNum == 6:
        block_length = 1024
        input_x = np.random.uniform(-1000, 1000, [block_length]).astype(input_type)
        golden = np.zeros(2).astype(output_type)
        it = np.argmin(input_x)
        idx = np.uint16(it)
        golden[0] = input_x[it]
        golden[1] = idx.view(np.float16)
    elif scenarioNum == 7:
        src_row = 13
        src_col = 57
        input_x = np.random.uniform(1, 10, [src_row, src_col]).astype(input_type)
        golden = np.zeros(src_row).astype(output_type)
        for i in range(src_row):
            golden[i] = np.sum(input_x[i, :])
    else:
        block_length = 1024
        repeat = block_length // one_repeat_items
        input_x = np.random.uniform(-1, 1, [block_length]).astype(input_type)
        golden = np.zeros(2 * repeat).astype(output_type)
        for i in range(repeat):
            golden[2 * i] = np.max(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
            max_index = np.argmax(input_x[i * one_repeat_items:(i + 1) * one_repeat_items])
            max_index = np.uint16(max_index)
            golden[2 * i + 1] = max_index.view(np.float16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenarioNum', type=int, default=1, choices=range(1, 8))
    args = parser.parse_args()
    gen_golden_data(args.scenarioNum)
