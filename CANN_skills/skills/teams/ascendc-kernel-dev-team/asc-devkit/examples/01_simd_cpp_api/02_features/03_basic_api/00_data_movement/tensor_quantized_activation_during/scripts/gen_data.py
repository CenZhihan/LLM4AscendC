#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import numpy as np
import tensorflow as tf
np.random.seed(19)


def ceil_div(a_value, b_value):
    """
    ceil division
    Parameters
    ----------
    a_value :operator
    b_value :division value

    Returns
    -------
    computational results
    """
    return (a_value + b_value - 1) // b_value


def clear_last_10bits_of_float32(fp32_data):
    int32_data = fp32_data.view("uint32")
    for index, data in enumerate(int32_data):
        int32_data[index] = np.bitwise_and(data, 0xFFFFE000)  # 1 sign bit, 8 exponent bits and 10 mantissa bits
    dealed_data = int32_data.view("float32")
    return dealed_data


def copy_conv2d(fm_shape, k_shape, fm_tensor_dtype, weight_tensor_dtype, output_l0c_dtype,
                deq_tensor_dtype, output_gm_dtype, stride_list, pad_list, pad_value, dilation_list,
                if_relu=False, deq_mode=None, deq_value=1.0, dst_src_stride=0, if_nz2nd=False,
                if_split=False, clip_relu=0, ele_wise=0):

    c1, h, w, c0 = fm_shape
    c1, kh, kw, cout, c0 = k_shape
    cin = c0 * c1
    cin_blocks = c1
    cout_blk = cout // 16
    stride_h, stride_w = stride_list
    dilation_h, dilation_w = dilation_list
    pad_left, pad_right, pad_top, pad_bot = pad_list
    kh_dilation = (kh - 1) * dilation_h + 1
    kw_dilation = (kw - 1) * dilation_w + 1
    ho = int(np.ceil((h + pad_top + pad_bot - kh_dilation + 1) / stride_h))
    wo = int(np.ceil((w + pad_right + pad_left - kw_dilation + 1) / stride_w))

    # mode 2
    if deq_tensor_dtype == np.float32:
        deq_tensor_value = np.random.uniform(-2, 2, (cout // (dst_src_stride + 1),)).astype("float32")
    elif deq_tensor_dtype == np.float16:
        deq_tensor_value = np.random.uniform(-2, 2, (16,)).astype("float16")
    else:
        temp_deq_tensor_value = np.random.uniform(-2, 2, (cout // (dst_src_stride + 1),)).astype("float32")
        deq_tensor_value = np.frombuffer(temp_deq_tensor_value, np.uint32)
        deq_tensor_value = deq_tensor_value.astype(np.uint64)
    if fm_tensor_dtype in ("float16", "float32"):
        fm = np.random.random(size=(1, h, w, c1 * c0)).astype(fm_tensor_dtype)
        weight = np.random.random(size=(kh, kw, c1 * c0, cout)).astype(weight_tensor_dtype)
    else:
        fm = np.random.randint(-5, 5, size=(1, h, w, c1 * c0)).astype(fm_tensor_dtype)
        weight = np.random.randint(-5, 5, size=(kh, kw, c1 * c0, cout)).astype(weight_tensor_dtype)
    if output_l0c_dtype not in ("float16", "float32"):
        # tensorflow mode
        with tf.compat.v1.Session() as sess:
            fm_padding = tf.pad(fm.astype("float32"), [[0, 0], [pad_top, pad_bot], [pad_left, pad_right], [0, 0]],
                                constant_values=np.float16(pad_value))
            ret = tf.nn.conv2d(fm_padding, weight.astype("float32"), (1, stride_list[0], stride_list[1], 1), 'VALID',
                               dilations=[1, dilation_h, dilation_w, 1]).eval()

            ret = ret.astype(output_l0c_dtype)
    else:
        # own way
        fm_padding = (np.ones((1, h + pad_top + pad_bot,
                               w + pad_left + pad_right, cin)) * pad_value).astype(fm_tensor_dtype)
        fm_padding[:, pad_top:pad_top + h, pad_left:pad_left + w, :] = fm

        round_howo = ceil_div(ho * wo, 16) * 16

        w_dilation = np.zeros((kh_dilation, kw_dilation, cin, cout), dtype=weight_tensor_dtype)
        for w_index in range(kw):
            for h_index in range(kh):
                if w_index == kw - 1:
                    w_new = kw_dilation - 1
                else:
                    w_new = w_index * dilation_w
                if h_index == kh - 1:
                    h_new = kh_dilation - 1
                else:
                    h_new = h_index * dilation_h
                w_dilation[h_new, w_new] = weight[h_index, w_index]

        load3d_fm = np.zeros((round_howo, cin * kh_dilation * kw_dilation), dtype=fm_tensor_dtype)
        load2d_w = np.zeros((cin * kh_dilation * kw_dilation, cout), dtype=weight_tensor_dtype)

        for channel_out in range(cout):
            filter_data = w_dilation[:, :, :, channel_out]
            load2d_w[:, channel_out] = filter_data.flatten()
        # ((kh_dilation, kw_dilation, c1, c0), Cout) -> ((c1, kh_dilation, kw_dilation, c0), Cout)
        load2d_w_copy = np.zeros((cin * kh_dilation * kw_dilation, cout), dtype=weight_tensor_dtype)
        for j in range(kh_dilation):
            for k in range(kw_dilation):
                for m in range(cin):
                    for n in range(cout):
                        index = j * kw_dilation * cin + k * cin + m
                        c0_index = index % c0
                        c1_index = (index // c0) % cin_blocks
                        kw_index = (index // cin) % kw_dilation
                        kh_index = (index // (cin * kw_dilation)) % kh_dilation
                        load2d_w_copy[c0_index + kw_index * c0 + kh_index * c0 * kw_dilation + \
                        c1_index * c0 * kh_dilation * kw_dilation, n] = load2d_w[index, n]

        channel_data = fm_padding[0, :, :, :]
        for r in range(ho):
            for c in range(wo):
                cur_input = channel_data[stride_h * r: stride_h * r + kh_dilation,
                            stride_w * c: kw_dilation + stride_w * c, :]
                load3d_fm[r * wo + c, :] = cur_input.flatten()
        # (round_howo, (kh_dilation, kw_dilation, c1, c0)) -> (round_howo, (c1, kh_dilation, kw_dilation, c0))
        load3d_fm_copy = np.zeros((round_howo, cin * kh_dilation * kw_dilation), dtype=fm_tensor_dtype)
        for n in range(round_howo):
            for j in range(kh_dilation):
                for k in range(kw_dilation):
                    for m in range(cin):
                        index = j * kw_dilation * cin + k * cin + m
                        c0_index = index % c0
                        c1_index = (index // c0) % cin_blocks
                        kw_index = (index // cin) % kw_dilation
                        kh_index = (index // (cin * kw_dilation)) % kh_dilation
                        load3d_fm_copy[n, c0_index + kw_index * c0 + kh_index * c0 * kw_dilation + \
                        c1_index * c0 * kh_dilation * kw_dilation] = load3d_fm[n, index]

        ret = np.zeros([1, round_howo, cout], dtype=output_l0c_dtype)
        howo_blk = round_howo // 16
        cout_blk = cout // 16
        for i in range(howo_blk):
            for q in range(cout_blk):
                for c1_index in range(cin_blocks):
                    for kh_index in range(kh_dilation):
                        for kw_index in range(kw_dilation):
                            j = kw_index + kh_index * kw_dilation + c1_index * kh_dilation * kw_dilation
                            part_fm = load3d_fm_copy[i * 16:i * 16 + 16, j * c0:j * c0 + c0].astype("float64")
                            part_w = load2d_w_copy[j * c0:j * c0 + c0, q * 16:q * 16 + 16].astype("float64")
                            conv_sum = np.matmul(part_fm, part_w)
                            ret[0, i * 16:i * 16 + 16, q * 16:q * 16 + 16] = \
                                (ret[0, i * 16:i * 16 + 16, q * 16:q * 16 + 16].astype("float64") + \
                                conv_sum.astype("float64")).astype(output_l0c_dtype)

    fm_input = np.zeros((c1, h, w, c0), dtype=fm_tensor_dtype)

    # note!! contains view
    weight_input = np.zeros((c1, kh, kw, cout, c0),
                            dtype=weight_tensor_dtype)

    for j in range(c1):
        for k in range(h):
            for m in range(w):
                for n in range(c0):
                    fm_input[j, k, m, n] = fm[0, k, m, j * c0 + n]

    for i in range(c1):
        for j in range(kh):
            for k in range(kw):
                for m in range(cout):
                    for n in range(c0):
                        weight_input[i, j, k, m, n] = \
                            weight[j, k, i * c0 + n, m]
    weight_input = weight_input.view(weight_tensor_dtype)

    if output_l0c_dtype not in ("float16", "float32"):
        if output_gm_dtype in ("int8", "uint8"):
            ret_z = np.zeros([cout_blk // 2, ho, wo, 32], dtype=output_l0c_dtype)
            for j in range(cout_blk // 2):
                for k in range(ho):
                    for m in range(wo):
                        for n in range(32):
                            ret_z[j, k, m, n] = ret[0, k, m, j * 32 + n]
        else:
            ret_z = np.zeros((cout_blk, ho, wo, 16), dtype=output_l0c_dtype)
            for j in range(cout_blk):
                for k in range(ho):
                    for m in range(wo):
                        for n in range(16):
                            ret_z[j, k, m, n] = ret[0, k, m, j * 16 + n]
    else:
        if output_gm_dtype in ("int8", "uint8"):
            ret_z = np.zeros([cout_blk // 2, ho, wo, 32], dtype=output_l0c_dtype)
            for j in range(cout_blk // 2):
                for k in range(ho):
                    for m in range(wo):
                        for n in range(32):
                            ret_z[j, k, m, n] = ret[0, k * wo + m, j * 32 + n]
        else:
            if if_split:
                new_cout_blk = cout // 8
                ret_z = np.zeros((new_cout_blk, ho, wo, 8), dtype=output_l0c_dtype)
                for j in range(new_cout_blk):
                    for k in range(ho):
                        for m in range(wo):
                            for n in range(8):
                                ret_z[j, k, m, n] = ret[0, k * wo + m, j * 8 + n]
            else:
                ret_z = np.zeros((cout_blk, ho, wo, 16), dtype=output_l0c_dtype)
                for j in range(cout_blk):
                    for k in range(ho):
                        for m in range(wo):
                            for n in range(16):
                                ret_z[j, k, m, n] = ret[0, k * wo + m, j * 16 + n]

    # notice: v220 do relu first and then do quant, but v100/v200 do quant first and then do relu
    if if_relu and deq_tensor_dtype == np.float32:
        ret_z = np.maximum(ret_z, 0)
    if if_relu and deq_tensor_dtype == np.uint64:
        ret_z = np.maximum(ret_z, 0)

    if deq_mode == "int322fp16":
        golden = ret_z.astype("float32")
        if isinstance(deq_value, (np.float32, float)):
            int32_data = np.int32(np.bitwise_and(np.float32(deq_value).view("int32"), 0xFFFFE000))
            deq_value = int32_data.view("float32")
            golden = golden[:] * np.float32(deq_value)
        else:
            if deq_tensor_dtype == np.float32:
                deq_tensor_value = clear_last_10bits_of_float32(deq_tensor_value)
                deq_tensor_value_fp32 = deq_tensor_value.astype("float32")
                golden = golden.transpose([1, 2, 0, 3])
                golden = golden.reshape([ho, wo, cout])
                for i in range(cout // (dst_src_stride + 1)):
                    golden[:, :, i * 16 * (dst_src_stride + 1):i * 16 * (dst_src_stride + 1) + 16] = \
                        golden[:, :, i * 16 * (dst_src_stride + 1):i * 16 * (dst_src_stride + 1) + 16] \
                        * deq_tensor_value_fp32[i * 16:(i + 1) * 16]
                golden = golden.reshape([ho, wo, cout // 16, 16])
                golden = golden.transpose([2, 0, 1, 3])
            elif deq_tensor_dtype == np.uint64:
                temp_deq_tensor_value = deq_tensor_value.astype(np.int32)
                temp_deq_tensor_value = clear_last_10bits_of_float32(temp_deq_tensor_value)
                deq_tensor_value_fp32 = temp_deq_tensor_value.astype("float32")
                golden = golden.transpose([1, 2, 0, 3])
                golden = golden.reshape([ho, wo, cout])
                for i in range(cout // (dst_src_stride + 1)):
                    golden[:, :, i * 16 * (dst_src_stride + 1):i * 16 * (dst_src_stride + 1) + 16] = \
                        golden[:, :, i * 16 * (dst_src_stride + 1):i * 16 * \
                        (dst_src_stride + 1) + 16] * deq_tensor_value_fp32[i * 16:(i + 1) * 16]
                golden = golden.reshape([ho, wo, cout // 16, 16])
                golden = golden.transpose([2, 0, 1, 3])
            else:
                golden = golden * deq_tensor_value
            ret_z = np.zeros((cout_blk, ho, wo, 16), dtype=golden.dtype)
            ret_z[::(dst_src_stride + 1), :, :, :] = golden[::(dst_src_stride + 1), :, :, :]
            golden = ret_z
        golden = golden.astype(output_gm_dtype)
    else:
        golden = ret_z.astype(output_gm_dtype)

    elewise_tensor = np.full((cout_blk, ho * wo, 16), 2).astype('float16')
    if if_relu and deq_tensor_dtype == np.float16:
        golden = np.maximum(golden, 0)
    if clip_relu == 1:
        golden = np.minimum(golden, 1)
    if ele_wise == 1:
        golden = golden + 2
    elif ele_wise == 2:
        golden = golden - 2
    if if_nz2nd:
        golden = golden.transpose((1, 2, 0, 3)).reshape(ho * wo, cout_blk * 16)
    return fm_input, weight_input, deq_tensor_value, elewise_tensor, golden



def copy_conv2d_gen_data(params):
    shape_fmi = params["fm_shape"]
    shape_weight = params["weight_shape"]
    stride_list = params["stride_list"]
    pad_list = params["pad_list"]
    pad_value = params["pad_value"]
    dilation_list = params["dilation_list"]

    dtype_fmi = params['fm_type']
    dtype_weight = params['weight_type']
    dtype_fmo = params['dst_gm_type']
    l0c_dtype = params['dst_l0c_type']
    deq_dtype = params['deq_dtype']

    deq_value = np.random.random(1).astype("float32")[0]
    deq_mode = None
    if params["quantize_params"] is not None:
        deq_mode = params["quantize_params"]["mode"]
        if params["quantize_params"]['mode_param'] is not None:
            deq_value = params["quantize_params"]["mode_param"]

    fmi, weight, deq_tensor_value, elewise_tensor, fmo = copy_conv2d(
        shape_fmi, shape_weight, dtype_fmi, dtype_weight, l0c_dtype, deq_dtype, dtype_fmo, stride_list=stride_list,
        pad_list=pad_list, pad_value=pad_value, dilation_list=dilation_list, if_relu=params['relu'],
        deq_mode=deq_mode, deq_value=deq_value, if_nz2nd=params['nz2nd'], if_split=params['channel_split'],
        clip_relu=params['clip_relu'], ele_wise=params['elewise_op'])
    return fmi, weight, deq_tensor_value, elewise_tensor, fmo


def gen_golden_data():
    my_params = {"fm_shape": [1, 4, 4, 32], "weight_shape": [1, 2, 2, 128, 32],
                "fm_type": np.int8, "weight_type": np.int8,
                "dst_l0c_type": "float32", "deq_dtype": np.uint64, "dst_gm_type": "half",
                "stride_list": [1, 1], "pad_list": [0, 0, 0, 0],
                "dilation_list": [1, 1], "pad_value": 0,
                "quantize_params": {"mode": "int322fp16", "mode_param": 1}, "kernel_name": "cce_copy_conv2d",
                "deq_type": "tensor", "relu": True, "nz2nd": True, "channel_split": False,
                "init_l1out": True, "clip_relu": 0, "ele_wise": 0, "elewise_op": 0}

    fm_data, we_data, deq_tensor_value, elewise_tensor, golden_data = copy_conv2d_gen_data(my_params)

    tiling = np.zeros([16]).astype(np.uint32)
    tiling[0] = 1
    tiling[1] = 4
    tiling[2] = 4
    tiling[3] = 2
    tiling[4] = 2
    tiling[5] = 128
    tiling[6] = 32
    tiling[7] = 1
    tiling[8] = 1
    tiling[9] = 10
    tiling[10] = True
    tiling[11] = True
    tiling[12] = False
    tiling[13] = 0
    tiling[14] = 0

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    fm_data.tofile("./input/fm_data.bin")
    we_data.tofile("./input/we_data.bin")
    deq_tensor_value.tofile("./input/deq_data.bin")
    elewise_tensor.tofile("./input/elewise_data.bin")
    tiling.tofile("./input/tiling_data.bin")
    golden_data.tofile("./output/golden_data.bin")


if __name__ == "__main__":
    gen_golden_data()