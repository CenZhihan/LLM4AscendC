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
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/asc_simd.h"

#define TEST_VECTOR_COMPUTE_CREATE_ITER_REG(class_name, c_api_name, cce_name)            \
                                                                                                \
class TestVectorCompute##class_name##_CApi : public testing::Test {                 \
protected:                                                                                      \
    void SetUp() {}                                                                             \
    void TearDown() {}                                                                          \
};                                                                                              \
                                                                                                \
namespace {                                                                                     \
iter_reg cce_name##_Stub(uint32_t offset) { return vector_address{}; }                          \
}                                                                                               \
                                                                                                \
TEST_F(TestVectorCompute##class_name##_CApi, c_api_name##_Succ)       \
{                                                                                               \
    uint32_t offset = 32;                                                                       \
    MOCKER_CPP(cce_name, iter_reg(uint32_t))                     \
        .times(1)                                                                               \
        .will(invoke(cce_name##_Stub));                                           \
                                                                                                \
    c_api_name(offset);                                                                \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

TEST_VECTOR_COMPUTE_CREATE_ITER_REG(Vag_b8, asc_create_iter_reg_b8, vag_b8);
TEST_VECTOR_COMPUTE_CREATE_ITER_REG(Vag_b16, asc_create_iter_reg_b16, vag_b16);
TEST_VECTOR_COMPUTE_CREATE_ITER_REG(Vag_b32, asc_create_iter_reg_b32, vag_b32);

#define TEST_VECTOR_COMPUTE_CREATE_ITER_REG_CUBE(class_name, c_api_name, cce_name)            \
                                                                                                \
class TestVectorCompute##class_name##_CApi_Cube : public testing::Test {                 \
protected:                                                                                      \
    void SetUp() {                                                                          \
        g_coreType = C_API_AIC_TYPE;                                                                          \
    }                                                                          \
    void TearDown() {                                                                          \
        g_coreType = C_API_AIV_TYPE;                                                                          \
    }                                                                          \
};                                                                                              \
                                                                                                \
                                                                                                \
TEST_F(TestVectorCompute##class_name##_CApi_Cube, c_api_name##_Succ)       \
{                                                                                               \
    uint32_t offset = 32;                                                                       \
    MOCKER_CPP(cce_name, iter_reg(uint32_t))                     \
        .times(0)                                                                               \
        .will(invoke(cce_name##_Stub));                                           \
                                                                                                \
    c_api_name(offset);                                                                \
    GlobalMockObject::verify();                                                                 \
}                                                                                               \

TEST_VECTOR_COMPUTE_CREATE_ITER_REG_CUBE(Vag_b8, asc_create_iter_reg_b8, vag_b8);
TEST_VECTOR_COMPUTE_CREATE_ITER_REG_CUBE(Vag_b16, asc_create_iter_reg_b16, vag_b16);
TEST_VECTOR_COMPUTE_CREATE_ITER_REG_CUBE(Vag_b32, asc_create_iter_reg_b32, vag_b32);