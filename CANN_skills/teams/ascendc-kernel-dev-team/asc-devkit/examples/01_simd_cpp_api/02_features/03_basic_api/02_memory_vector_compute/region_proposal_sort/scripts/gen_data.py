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


def gen_golden_data():
    """
    生成Region Proposal排序流水线的测试数据和Golden数据
    输入：[1, 64]个half score值
    输出：[1, 64]个half score值（按降序排列）
    流水线：ProposalConcat → RpSort16 → MrgSort4 → ProposalExtract
    """
    score_count = 64
    input_type = np.dtype("float16")

    src = np.random.uniform(-1000, 1000, [1, score_count]).astype(input_type)

    # 全局降序排序作为golden
    golden = np.sort(src.flatten())[::-1].astype(input_type).reshape(1, -1)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    src.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
