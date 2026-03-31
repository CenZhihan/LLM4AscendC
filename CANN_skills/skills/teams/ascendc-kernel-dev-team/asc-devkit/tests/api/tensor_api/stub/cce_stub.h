/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TESTS_API_TENSOR_API_STUB_CCE_STUB_H
#define TESTS_API_TENSOR_API_STUB_CCE_STUB_H

#include <cstdint>
#include "stub_fun.h"

#ifndef __biasbuf__
#define __biasbuf__
#endif

static bool is_mock_copy_matrix_cc_to_gm = false;
static uint16_t n_size_global = 0;
static uint16_t m_size_global = 0;
static uint32_t dst_stride_global = 0;
static uint16_t src_stride_global = 0;
static bool NZ2ND_en_global = false;
static bool NZ2DN_en_global = false;
static void* gm_addr_global = nullptr;
static uint64_t quant_pre_global = 0;

#define mock_copy_matrix_cc_to_gm(DstT, L0cT) \
inline void copy_matrix_cc_to_gm( \
    __gm__ DstT* dst_addr, __cc__ L0cT* src_addr, uint8_t sid, uint16_t n_size, uint16_t m_size, \
    uint32_t loop_dst_stride, uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre, \
    uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre, bool split_en, bool NZ2ND_en, \
    uint64_t quant_post, uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op, \
    bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en) {\
        if (is_mock_copy_matrix_cc_to_gm) { \
            EXPECT_EQ(n_size, n_size_global); \
            EXPECT_EQ(m_size, m_size_global); \
            EXPECT_EQ(loop_dst_stride, dst_stride_global); \
            EXPECT_EQ(loop_src_stride, src_stride_global); \
            EXPECT_EQ(NZ2ND_en, NZ2ND_en_global); \
            EXPECT_EQ(NZ2DN_en, NZ2DN_en_global); \
            EXPECT_EQ(dst_addr, gm_addr_global); \
            EXPECT_EQ(quant_pre, quant_pre_global); \
        } \
    } \

mock_copy_matrix_cc_to_gm(float, float);
mock_copy_matrix_cc_to_gm(half, float);
mock_copy_matrix_cc_to_gm(int8_t, int32_t);

#endif
