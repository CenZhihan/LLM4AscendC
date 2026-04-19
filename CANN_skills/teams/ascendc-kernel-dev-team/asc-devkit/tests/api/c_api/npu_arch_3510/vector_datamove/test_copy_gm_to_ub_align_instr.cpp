/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

#define TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(class_name, c_api_name, cce_name, dst_date_type, src_date_type)      \
                                                                                                \
class TestVectorDatamove##class_name##_##dst_date_type##_CApi : public testing::Test {                                   \
protected:                                                                                      \
    void SetUp() {}                                                                             \
    void TearDown() {}                                                                          \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
                                                                                                \
void cce_name##_##dst_date_type##_##src_date_type##_uint64_t_Stub(__ubuf__ dst_date_type *dst, __gm__ src_date_type *src, uint8_t sid, uint32_t n_burst, uint32_t len_burst, \
            uint8_t left_padding_num, uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride, uint32_t dst_stride) \
{                                                                                               \
    EXPECT_EQ(dst, reinterpret_cast<__ubuf__ void *>(11));                                      \
    EXPECT_EQ(src, reinterpret_cast<__gm__ void *>(22));                                        \
    EXPECT_EQ(sid, static_cast<uint8_t>(0));                                                    \
    EXPECT_EQ(n_burst, static_cast<uint32_t>(1));                                               \
    EXPECT_EQ(len_burst, static_cast<uint32_t>(44));                                             \
    EXPECT_EQ(left_padding_num, static_cast<uint8_t>(0));                                       \
    EXPECT_EQ(right_padding_num, static_cast<uint8_t>(0));                                      \
    EXPECT_EQ(enable_constant_pad, false);                                               \
    EXPECT_EQ(l2_cache_mode, static_cast<uint8_t>(0));                                               \
    EXPECT_EQ(src_stride, static_cast<uint64_t>(0));                                               \
    EXPECT_EQ(dst_stride, static_cast<uint32_t>(0));                                               \
}                                                                                               \
                                                                                                \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorDatamove##class_name##_##dst_date_type##_CApi, c_api_name##_CopyConfig_Succ)                            \
{                                                                                               \
    __ubuf__ dst_date_type *dst = reinterpret_cast<__ubuf__ dst_date_type *>(11);               \
    __gm__ src_date_type *src = reinterpret_cast<__gm__ src_date_type *>(22);                   \
                                                                                                \
    uint32_t n_burst = static_cast<uint64_t>(1);                                                \
    uint32_t len_burst = static_cast<uint64_t>(44);                                              \
    uint8_t left_padding_num = static_cast<uint64_t>(0);                                        \
    uint8_t right_padding_num = static_cast<uint64_t>(0);                                       \
    bool enable_constant_pad = false;                                                \
    uint8_t l2_cache_mode = static_cast<uint8_t>(0);                                                \
    uint64_t src_gap = static_cast<uint64_t>(0);                                                \
    uint32_t dst_gap = static_cast<uint64_t>(0);                                                \
                                                                                                \
    MOCKER_CPP(cce_name, void(__ubuf__ dst_date_type *, __gm__ src_date_type *, uint8_t, uint32_t, uint32_t, uint8_t, uint8_t, bool, uint8_t, uint64_t, uint32_t))\
            .times(1)                                                                           \
            .will(invoke(cce_name##_##dst_date_type##_##src_date_type##_uint64_t_Stub));                                            \
                                                                                                \
    c_api_name(dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode, src_gap, dst_gap); \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, int8_t, int8_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, int16_t, int16_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, int32_t, int32_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, half, half);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, float, float);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_COPY_UBUF_TO_GM_ALIGN_INSTR(CopyGM2UBALIGN, asc_copy_gm2ub_align, copy_gm_to_ubuf_align_v2, fp8_e4m3fn_t, fp8_e4m3fn_t);