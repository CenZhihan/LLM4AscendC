/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class TestMrgsort4CAPI : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define TEST_MRGSORT4_PARAMS(type)                                                                                     \
    namespace {                                                                                                        \
    void vmrgsort4_##type##_params_Stub(__ubuf__ type* dst, __ubuf__ type* src, uint8_t repeat,                        \
                                        uint16_t element_length_0, uint16_t element_length_1,                          \
                                        uint16_t element_length_2, uint16_t element_length_3,                          \
                                        bool if_exhausted_suspension, uint8_t valid_bit)                               \
    {                                                                                                                  \
        EXPECT_EQ(dst, reinterpret_cast<__ubuf__ type*>(1));                                                           \
        EXPECT_EQ(src, reinterpret_cast<__ubuf__ type*>(2));                                                           \
        EXPECT_EQ(repeat, static_cast<uint8_t>(3));                                                                    \
        EXPECT_EQ(element_length_0, static_cast<uint16_t>(4));                                                         \
        EXPECT_EQ(element_length_1, static_cast<uint16_t>(5));                                                         \
        EXPECT_EQ(element_length_2, static_cast<uint16_t>(6));                                                         \
        EXPECT_EQ(element_length_3, static_cast<uint16_t>(7));                                                         \
        EXPECT_EQ(if_exhausted_suspension, true);                                                                      \
        EXPECT_EQ(valid_bit, static_cast<uint8_t>(8));                                                                 \
    }                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestMrgsort4CAPI, asc_mrgsort4_##type##_params_Succ)                                                        \
    {                                                                                                                  \
        __ubuf__ type* dst = reinterpret_cast<__ubuf__ type*>(1);                                                      \
        __ubuf__ type* src = reinterpret_cast<__ubuf__ type*>(2);                                                      \
        uint8_t repeat = 3;                                                                                            \
        uint16_t element_length_0 = 4;                                                                                 \
        uint16_t element_length_1 = 5;                                                                                 \
        uint16_t element_length_2 = 6;                                                                                 \
        uint16_t element_length_3 = 7;                                                                                 \
        bool if_exhausted_suspension = true;                                                                           \
        uint8_t valid_bit = 8;                                                                                         \
                                                                                                                       \
        MOCKER_CPP(vmrgsort4, void(__ubuf__ type*, __ubuf__ type*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t,    \
                                   bool, uint8_t))                                                                     \
            .times(1)                                                                                                  \
            .will(invoke(vmrgsort4_##type##_params_Stub));                                                             \
                                                                                                                       \
        asc_mrgsort4(dst, src, repeat, element_length_0, element_length_1, element_length_2, element_length_3,         \
                     if_exhausted_suspension, valid_bit);                                                              \
        GlobalMockObject::verify();                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    TEST_F(TestMrgsort4CAPI, asc_mrgsort4_sync_##type##_params_Succ)                                                   \
    {                                                                                                                  \
        __ubuf__ type* dst = reinterpret_cast<__ubuf__ type*>(1);                                                      \
        __ubuf__ type* src = reinterpret_cast<__ubuf__ type*>(2);                                                      \
        uint8_t repeat = 3;                                                                                            \
        uint16_t element_length_0 = 4;                                                                                 \
        uint16_t element_length_1 = 5;                                                                                 \
        uint16_t element_length_2 = 6;                                                                                 \
        uint16_t element_length_3 = 7;                                                                                 \
        bool if_exhausted_suspension = true;                                                                           \
        uint8_t valid_bit = 8;                                                                                         \
                                                                                                                       \
        MOCKER_CPP(vmrgsort4, void(__ubuf__ type*, __ubuf__ type*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t,    \
                                   bool, uint8_t))                                                                     \
            .times(1)                                                                                                  \
            .will(invoke(vmrgsort4_##type##_params_Stub));                                                             \
                                                                                                                       \
        asc_mrgsort4_sync(dst, src, repeat, element_length_0, element_length_1, element_length_2, element_length_3,    \
                          if_exhausted_suspension, valid_bit);                                                         \
        GlobalMockObject::verify();                                                                                    \
    }

TEST_MRGSORT4_PARAMS(half)
TEST_MRGSORT4_PARAMS(float)
