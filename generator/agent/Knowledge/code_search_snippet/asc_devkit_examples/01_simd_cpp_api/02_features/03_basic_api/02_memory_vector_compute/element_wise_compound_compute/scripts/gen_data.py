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
import argparse
import numpy as np

np.random.seed(9)


def gen_golden_data(scenario=2):
    """
    根据场景编号生成输入数据和Golden数据
    
    Args:
        scenario: 场景编号
            1 - CastDequant: int16_t -> uint8_t 反量化
            2 - AddRelu: half -> half 加法与ReLU激活
            3 - Axpy: half -> half 标量乘法与向量加法
    """
    data_size = 512

    if scenario == 1:
        # 场景1：CastDequant - int16_t输入，uint8_t输出
        # CastDequant halfBlock=false时：每16个int16对应32字节输出（前16字节有效）
        # DATA_SIZE=512个int16，共32组，输出1024字节
        input0 = np.random.randint(0, 256, size=data_size).astype(np.int16)
        input1 = np.zeros(data_size, dtype=np.int16)  # 占位，不使用

        # Golden生成：每16个int16为一组，转换后放入32字节的输出位置（前16字节有效）
        golden = np.zeros(data_size * 2, dtype=np.uint8)
        for i in range(data_size // 16):  # 32组
            golden[i * 32:i * 32 + 16] = input0[i * 16:i * 16 + 16].astype(np.uint8)

    elif scenario == 2:
        # 场景2：AddRelu - half输入输出
        # 计算公式：dst = max(src0 + src1, 0)
        input0 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
        input1 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
        golden = np.maximum(input0 + input1, 0).astype(np.float16)

    elif scenario == 3:
        # 场景3：Axpy - half输入输出
        # 计算公式：dst = dst + src * scalar，scalar=2.0
        # input0为src，input1为初始dst
        input0 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
        input1 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
        scalar = 2.0
        golden = (input1 + input0 * scalar).astype(np.float16)

    else:
        raise ValueError(f"Invalid scenario: {scenario}, must be 1, 2, or 3")

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input0.tofile("./input/input0.bin")
    input1.tofile("./input/input1.bin")
    golden.tofile("./output/golden.bin")

    print(f"Generated data for scenario {scenario}: input0 shape={input0.shape}, dtype={input0.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scenario', type=int, default=2, choices=[1, 2, 3],
                        help='Scenario number: 1=CastDequant, 2=AddRelu, 3=Axpy')
    args = parser.parse_args()
    gen_golden_data(args.scenario)