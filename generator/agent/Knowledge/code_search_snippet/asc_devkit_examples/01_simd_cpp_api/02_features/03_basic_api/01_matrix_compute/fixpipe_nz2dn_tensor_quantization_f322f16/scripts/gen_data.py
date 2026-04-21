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

np.random.seed(19)
BLOCK_SIZE = 16


def ceil_div(a_value, b_value):
    return (a_value + b_value - 1) // b_value


def tik_conv2d(fm_shape, k_shape, fm_tensor_dtype, weight_tensor_dtype, output_l0c_dtype, output_gm_dtype,
                stride_list, pad_list, pad_value, dilation_list, dst_src_stride=0):
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
    temp_deq_tensor_value = np.random.uniform(-2, 2, (cout // (dst_src_stride + 1),)).astype("float32")
    deq_tensor_value = np.frombuffer(temp_deq_tensor_value, np.uint32)
    deq_tensor_value = deq_tensor_value.astype(np.uint64)

    fm = np.random.randint(-5, 5, size=(1, h, w, c1 * c0)).astype(fm_tensor_dtype)
    weight = np.random.randint(-5, 5, size=(kh, kw, c1 * c0, cout)).astype(weight_tensor_dtype)

    fm_padding = (np.ones((1, h + pad_top + pad_bot,
                        w + pad_left + pad_right, cin)) * pad_value).astype(fm_tensor_dtype)
    fm_padding[:, pad_top:pad_top + h, pad_left:pad_left + w, :] = fm
    round_howo = ceil_div(ho * wo, 16) * 16

    w_dilation = np.zeros((kh_dilation, kw_dilation, cin, cout), dtype=weight_tensor_dtype)

    h_indices, w_indices = np.meshgrid(np.arange(kh), np.arange(kw), indexing='ij')

    h_new = np.where(h_indices == kh - 1, kh_dilation - 1, h_indices * dilation_h)
    w_new = np.where(w_indices == kw - 1, kw_dilation - 1, w_indices * dilation_w)

    w_dilation[h_new, w_new] = weight[h_indices, w_indices]

    load3d_fm = np.zeros((round_howo, cin * kh_dilation * kw_dilation), dtype=fm_tensor_dtype)
    load2d_w = np.zeros((cin * kh_dilation * kw_dilation, cout), dtype=weight_tensor_dtype)

    for channel_out in range(cout):
        filter_data = w_dilation[:, :, :, channel_out]
        load2d_w[:, channel_out] = filter_data.flatten()
    load2d_w_tik = np.zeros((cin * kh_dilation * kw_dilation, cout), dtype=weight_tensor_dtype)

    source_indices = np.arange(kh_dilation * kw_dilation * cin)

    c0_index = source_indices % c0
    c1_index = (source_indices // c0) % cin_blocks
    kw_index = (source_indices // cin) % kw_dilation
    kh_index = (source_indices // (cin * kw_dilation)) % kh_dilation

    new_index = (
        c0_index + 
        kw_index * c0 + 
        kh_index * c0 * kw_dilation + 
        c1_index * c0 * kh_dilation * kw_dilation
    )

    load2d_w_tik = np.zeros_like(load2d_w)

    np.add.at(load2d_w_tik, new_index, load2d_w[source_indices])
                    
    channel_data = fm_padding[0, :, :, :]
    for r in range(ho):
        for c in range(wo):
            cur_input = channel_data[stride_h * r: stride_h * r + kh_dilation,
                        stride_w * c:kw_dilation + stride_w * c, :]
            load3d_fm[r * wo + c, :] = cur_input.flatten()
    load3d_fm_tik = np.zeros((round_howo, cin * kh_dilation * kw_dilation), dtype=fm_tensor_dtype)

    source_col_indices = np.arange(kh_dilation * kw_dilation * cin)

    c0_index = source_col_indices % c0
    c1_index = (source_col_indices // c0) % cin_blocks
    kw_index = (source_col_indices // cin) % kw_dilation
    kh_index = (source_col_indices // (cin * kw_dilation)) % kh_dilation

    new_col_indices = (
        c0_index + 
        kw_index * c0 + 
        kh_index * c0 * kw_dilation + 
        c1_index * c0 * kh_dilation * kw_dilation
    )

    load3d_fm_tik = np.zeros_like(load3d_fm)

    row_indices = np.arange(round_howo)[:, np.newaxis]
    np.add.at(load3d_fm_tik, (row_indices, new_col_indices), load3d_fm[row_indices, source_col_indices])

    ret = np.zeros([1, round_howo, cout], dtype=output_l0c_dtype)
    howo_blk = round_howo // 16
    cout_blk = cout // 16

    total_cin_blocks = cin_blocks * kh_dilation * kw_dilation

    fm_block_struct = load3d_fm_tik.reshape(howo_blk, BLOCK_SIZE, total_cin_blocks, c0)

    w_block_struct = load2d_w_tik.reshape(total_cin_blocks, c0, cout_blk, BLOCK_SIZE)

    for i in range(howo_blk):
        for q in range(cout_blk):
            fm_chunk = fm_block_struct[i, :, :, :]
            
            w_chunk = w_block_struct[:, :, q, :]
            
            fm_chunk_aligned = fm_chunk.transpose(1, 0, 2)
            
            partial_sums = np.matmul(fm_chunk_aligned, w_chunk)
            
            conv_sum = np.sum(partial_sums, axis=0)
            
            fm_slice = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
            w_slice = slice(q * BLOCK_SIZE, (q + 1) * BLOCK_SIZE)
            ret[0, fm_slice, w_slice] = (
                ret[0, fm_slice, w_slice].astype("float64") +
                conv_sum.astype("float64")
            ).astype(output_l0c_dtype)
    
                      
    fm_input = np.zeros((c1, h, w, c0), dtype=fm_tensor_dtype)
    weight_input = np.zeros((c1, kh, kw, cout, c0), dtype=weight_tensor_dtype)

    fm_squeezed = fm[0]

    fm_reshaped = fm_squeezed.reshape(h, w, c1, c0)

    fm_input = np.transpose(fm_reshaped, (2, 0, 1, 3))
    
    weight_reshaped = weight.reshape(kh, kw, c1, c0, cout)

    weight_input = weight_reshaped.transpose(2, 0, 1, 4, 3)

    j_indices, k_indices, m_indices, n_indices = np.meshgrid(
        np.arange(cout_blk),
        np.arange(ho),
        np.arange(wo),
        np.arange(16),
        indexing='ij'
    )

    source_row_indices = k_indices * wo + m_indices
    source_col_indices = j_indices * 16 + n_indices

    ret_z = np.zeros((cout_blk, ho, wo, 16), dtype=output_l0c_dtype)
    ret_z[j_indices, k_indices, m_indices, n_indices] = ret[0, source_row_indices, source_col_indices]

    ret_z = np.maximum(ret_z, 0)
    golden = ret_z.astype("float32")
    golden = golden.astype(output_gm_dtype)
    golden = golden.transpose((1, 2, 0, 3)).reshape(ho * wo, cout_blk * 16).transpose(1, 0)
    return fm_input, weight_input, deq_tensor_value, golden


class QuantMode:
    F322F16 = 1


def gen_golden_data():
    my_params = {"fm_shape": [2, 4, 4, 16], "weight_shape": [2, 2, 2, 128, 16],
                "fm_type": "half", "weight_type": "half",
                "dst_l0c_type": "float32", "deq_dtype": "uint64_t", "dst_gm_type": "half",
                "stride_list": [1, 1], "pad_list": [0, 0, 0, 0],
                "dilation_list": [2, 2], "pad_value": 0}
    
    fm_data, we_data, deq_tensor_value, golden_data = tik_conv2d(
        my_params["fm_shape"], my_params["weight_shape"], my_params["fm_type"], my_params["weight_type"],
        my_params["dst_l0c_type"], my_params["dst_gm_type"], stride_list=my_params["stride_list"],
        pad_list=my_params["pad_list"], pad_value=my_params["pad_value"], dilation_list=my_params["dilation_list"])

    tiling = np.zeros(16, dtype=np.uint32)
    tiling[0] = 2
    tiling[1] = 4
    tiling[2] = 4
    tiling[3] = 2
    tiling[4] = 2
    tiling[5] = 128
    tiling[6] = 16
    tiling[7] = 2
    tiling[8] = 2
    tiling[9] = 1
    tiling[10] = True
    tiling[11] = 2
 
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    fm_data.tofile("./input/fm_data.bin")
    we_data.tofile("./input/we_data.bin")
    deq_tensor_value.tofile("./input/deq_tensor.bin")
    tiling.tofile("./input/tiling.bin")
    golden_data.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data()