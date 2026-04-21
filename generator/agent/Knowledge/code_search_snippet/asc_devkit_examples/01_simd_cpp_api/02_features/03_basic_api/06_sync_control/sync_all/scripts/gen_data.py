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
    input_type = np.float32
    work_type = input_type
    sync_type = np.int32

    num_blocks = 8
    sync_need_size = 8
    block_length = 256
    scalar_value = 2
    input_shape = [block_length]
    input_x = np.ones(input_shape).astype(input_type)
    sync = np.zeros(num_blocks * sync_need_size).astype(sync_type)
    work = np.zeros(input_shape).astype(work_type)
    golden = input_x * scalar_value * num_blocks
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    work.tofile("./input/input_work.bin")
    sync.tofile("./input/input_sync.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
