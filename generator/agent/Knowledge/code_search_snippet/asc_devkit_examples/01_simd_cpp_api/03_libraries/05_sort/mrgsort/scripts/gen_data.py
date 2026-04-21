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
    element_count = 128
    src_value = np.arange(0, element_count, 1, np.float32)
    np.random.shuffle(src_value)

    src_index = np.arange(0, element_count, 1, np.uint32)

    sorted_index_golden = np.argsort(src_value)[::-1].astype(np.uint32)
    sorted_value_golden = src_value[sorted_index_golden]
    
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    src_value.tofile("./input/src_value.bin")
    src_index.tofile("./input/src_index.bin")
    
    sorted_value_golden.tofile("./output/golden_output_dstValue.bin")
    sorted_index_golden.tofile("./output/golden_output_dstIndex.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
