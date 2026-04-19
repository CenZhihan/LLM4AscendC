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


def get_data():
    x = np.ones([1024]).astype(np.float32)
    y = np.ones([1024]).astype(np.float32)
    z = x + y
    return x, y, z


def gen_golden_data_simple():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x, y, z = get_data()
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    z.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
