/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
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

#define TEST_CUBE_COMPUTE_MMAD_INSTR(c_type, a_type, b_type, type_prefix)                                              \
                                                                                                                       \
    class TestMmad##type_prefix##CAPI : public testing::Test {                                                         \
    protected:                                                                                                         \
        void SetUp()                                                                                                   \
        {                                                                                                              \
            g_coreType = C_API_AIC_TYPE;                                                                               \
        }                                                                                                              \
        void TearDown()                                                                                                \
        {                                                                                                              \
            g_coreType = C_API_AIV_TYPE;                                                                               \
        }                                                                                                              \
    };                                                                                                                 \
                                                                                                                       \
    namespace {                                                                                                        \
    void mad_##type_prefix##_params_Stub(__cc__ c_type* c, __ca__ a_type* a, __cb__ b_type* b, uint16_t m, uint16_t k, \
                                         uint16_t n, uint8_t unit_flag_ctrl, bool gemv_ctrl, bool btbuf_ctrl,          \
                                         bool zero_cmatrix_ctrl)                                                       \
    {                                                                                                                  \
        EXPECT_EQ(c, reinterpret_cast<__cc__ c_type*>(1));                                                             \
        EXPECT_EQ(a, reinterpret_cast<__ca__ a_type*>(2));                                                             \
        EXPECT_EQ(b, reinterpret_cast<__cb__ b_type*>(3));                                                             \
        EXPECT_EQ(m, static_cast<uint16_t>(4));                                                                        \
        EXPECT_EQ(k, static_cast<uint16_t>(5));                                                                        \
        EXPECT_EQ(n, static_cast<uint16_t>(6));                                                                        \
        EXPECT_EQ(unit_flag_ctrl, static_cast<uint8_t>(1));                                                            \
        EXPECT_EQ(gemv_ctrl, true);                                                                                    \
        EXPECT_EQ(btbuf_ctrl, false);                                                                                  \
        EXPECT_EQ(zero_cmatrix_ctrl, true);                                                                            \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestMmad##type_prefix##CAPI, mmad_##type_prefix##_params_Succ)                                              \
    {                                                                                                                  \
        __cc__ c_type* c = reinterpret_cast<__cc__ c_type*>(1);                                                        \
        __ca__ a_type* a = reinterpret_cast<__ca__ a_type*>(2);                                                        \
        __cb__ b_type* b = reinterpret_cast<__cb__ b_type*>(3);                                                        \
        uint16_t m = 4;                                                                                                \
        uint16_t k = 5;                                                                                                \
        uint16_t n = 6;                                                                                                \
        uint8_t unit_flag_ctrl = 1;                                                                                    \
        bool gemv_ctrl = true;                                                                                         \
        bool btbuf_ctrl = false;                                                                                       \
        bool zero_cmatrix_ctrl = true;                                                                                 \
                                                                                                                       \
        MOCKER_CPP(mad, void(__cc__ c_type * c_matrix, __ca__ a_type * a_matrix, __cb__ b_type * b_matrix, uint16_t m, \
                             uint16_t k, uint16_t n, uint8_t unit_flag_ctrl, bool gemv_ctrl, bool btbuf_ctrl,          \
                             bool zero_cmatrix_ctrl))                                                                  \
            .times(1)                                                                                                  \
            .will(invoke(mad_##type_prefix##_params_Stub));                                                            \
                                                                                                                       \
        asc_mmad(c, a, b, m, k, n, unit_flag_ctrl, gemv_ctrl, btbuf_ctrl, zero_cmatrix_ctrl);                          \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestMmad##type_prefix##CAPI, mmad_sync_##type_prefix##_params_Succ)                                         \
    {                                                                                                                  \
        __cc__ c_type* c = reinterpret_cast<__cc__ c_type*>(1);                                                        \
        __ca__ a_type* a = reinterpret_cast<__ca__ a_type*>(2);                                                        \
        __cb__ b_type* b = reinterpret_cast<__cb__ b_type*>(3);                                                        \
        uint16_t m = 4;                                                                                                \
        uint16_t k = 5;                                                                                                \
        uint16_t n = 6;                                                                                                \
        uint8_t unit_flag_ctrl = 1;                                                                                    \
        bool gemv_ctrl = true;                                                                                         \
        bool btbuf_ctrl = false;                                                                                       \
        bool zero_cmatrix_ctrl = true;                                                                                 \
                                                                                                                       \
        MOCKER_CPP(mad, void(__cc__ c_type * c_matrix, __ca__ a_type * a_matrix, __cb__ b_type * b_matrix, uint16_t m, \
                             uint16_t k, uint16_t n, uint8_t unit_flag_ctrl, bool gemv_ctrl, bool btbuf_ctrl,          \
                             bool zero_cmatrix_ctrl))                                                                  \
            .times(1)                                                                                                  \
            .will(invoke(mad_##type_prefix##_params_Stub));                                                            \
                                                                                                                       \
        asc_mmad_sync(c, a, b, m, k, n, unit_flag_ctrl, gemv_ctrl, btbuf_ctrl, zero_cmatrix_ctrl);                     \
        GlobalMockObject::verify();                                                                                    \
    }

TEST_CUBE_COMPUTE_MMAD_INSTR(float, bfloat16_t, bfloat16_t, bfloat16);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, float8_e4m3_t, float8_e4m3_t, e4m3_e4m3);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, float8_e4m3_t, float8_e5m2_t, e4m3_e5m2);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, float8_e5m2_t, float8_e4m3_t, e5m2_e4m3);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, float8_e5m2_t, float8_e5m2_t, e5m2_e5m2);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, half, half, half);
TEST_CUBE_COMPUTE_MMAD_INSTR(float, float, float, float);
TEST_CUBE_COMPUTE_MMAD_INSTR(int32_t, int8_t, int8_t, int8);
