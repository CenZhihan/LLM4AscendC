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
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置tensorflow日志等级


def gen_golden_data():
    c1, h, w, c0 = 2, 4, 4, 16
    kh, kw = 2, 2
    cout = 16
    dilation_h, dilation_w = 2, 2
    pad_top, pad_bottom, pad_left, pad_right = 1, 1, 1, 1
    stride_h, stride_w = 1, 1
    ho = (h + pad_top + pad_bottom - dilation_h * (kh - 1) - 1) / stride_h + 1
    wo = (w + pad_left + pad_right - dilation_w * (kw - 1) - 1) / stride_w + 1

    fm = np.random.randint(1, 10, [1, h, w, c1 * c0]).astype(np.float16)
    weight = np.random.randint(1, 10, [kh, kw, c1 * c0, cout]).astype(np.float16)
    with tf.compat.v1.Session():
        fm_padding = tf.pad(fm.astype("float32"), [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                            constant_values=np.float16(0))
        golden = tf.nn.conv2d(fm_padding, weight.astype("float32"), (1, stride_h, stride_w, 1), "VALID",
                     dilations=[1, dilation_h, dilation_w, 1]).eval()
    golden = golden.astype(np.float32)

    fm_gm = np.zeros((c1, h, w, c0), dtype=np.float16)
    for a in range(c1):
        for b in range(h):
            for c in range(w):
                for d in range(c0):
                    fm_gm[a, b, c, d] = fm[0, b, c, a * c0 + d]

    weight_gm = np.zeros((c1, kh, kw, cout, c0), dtype=np.float16)
    for a in range(c1):
        for b in range(kh):
            for c in range(kw):
                for d in range(cout):
                    for e in range(c0):
                        weight_gm[a, b, c, d, e] = weight[b, c, a * c0 + e, d]

    os.system("mkdir -p input")
    os.system("mkdir -p output")

    fm_gm.tofile("./input/x1_gm.bin")
    weight_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()