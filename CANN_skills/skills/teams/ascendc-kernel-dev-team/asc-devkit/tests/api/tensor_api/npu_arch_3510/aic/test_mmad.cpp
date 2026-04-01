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
#include "tensor_api/stub/cce_stub.h"
#include <mockcpp/mockcpp.hpp>
#include "include/experimental/tensor_api/tensor.h"

class Tensor_Api_Mmad : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};


#define MMAD_TEST_BASE(DST_TYPE, SRC_TYPE, M, N, K, K_ALIGN, C_SOURCE, MOCK_FUNC) \
uint8_t A2##DST_TYPE##_##SRC_TYPE[256 * 256 * sizeof(SRC_TYPE)] = {0};\
uint8_t B2##DST_TYPE##_##SRC_TYPE[256 * 256 * sizeof(SRC_TYPE)] = {0};\
uint8_t C2##DST_TYPE##_##SRC_TYPE[256 * 256 * sizeof(DST_TYPE)] = {0};\
void MmadTest##DST_TYPE##_##SRC_TYPE(__cc__ DST_TYPE* dst, __ca__ SRC_TYPE* fm, __cb__ SRC_TYPE* filter, uint16_t m, uint16_t k, uint16_t n,\
        uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal)\
{\
    using namespace AscendC::Te;\
    EXPECT_EQ(dst, reinterpret_cast<__cc__ DST_TYPE*>(C2##DST_TYPE##_##SRC_TYPE));\
    EXPECT_EQ(fm, reinterpret_cast<__ca__ SRC_TYPE*>(A2##DST_TYPE##_##SRC_TYPE));\
    EXPECT_EQ(filter, reinterpret_cast<__cb__ SRC_TYPE*>(B2##DST_TYPE##_##SRC_TYPE));\
    EXPECT_EQ(m, M); EXPECT_EQ(n, N); EXPECT_EQ(k, K);\
    EXPECT_EQ(unitFlag, 0);\
    EXPECT_EQ(kDirectionAlign, kDirectionAlign);\
    EXPECT_EQ(cmatrixSource, cmatrixSource);\
    EXPECT_EQ(cmatrixInitVal, true);\
}\
\
TEST_F(Tensor_Api_Mmad, MmadOperation##MOCK_FUNC##_##DST_TYPE##_##SRC_TYPE##_##M##_##N##_##K)\
{\
    using namespace AscendC::Te;\
    auto a2Addr = reinterpret_cast<__ca__ SRC_TYPE*>(A2##DST_TYPE##_##SRC_TYPE);\
    auto l0aTensor = MakeTensor(MakeL0AmemPtr(a2Addr), MakeNzLayout<SRC_TYPE>(M, K));\
\
    auto b2Addr = reinterpret_cast<__cb__ SRC_TYPE*>(B2##DST_TYPE##_##SRC_TYPE);\
    auto l0bTensor = MakeTensor(MakeL0BmemPtr(b2Addr), MakeZnLayout<SRC_TYPE>(K, N));\
\
    auto c2Addr = reinterpret_cast<__cc__ DST_TYPE*>(C2##DST_TYPE##_##SRC_TYPE);\
    auto l0cTensor = MakeTensor(MakeL0CmemPtr(c2Addr), MakeNzLayout<AscendC::Std::ignore_t>(M, N));\
\
    MOCKER_CPP(MOCK_FUNC, void(__cc__ DST_TYPE *, __ca__ SRC_TYPE *,\
                __cb__ SRC_TYPE *, uint16_t, uint16_t, uint16_t, uint8_t,\
                bool, bool, bool))\
            .times(1)\
            .will(invoke(MmadTest##DST_TYPE##_##SRC_TYPE));\
    MmadParams para;\
    para.m = M;\
    para.n = N;\
    para.k = K;\
    para.unitFlag = 0;\
    para.cmatrixInitVal = true;\
    Mmad(l0cTensor, l0aTensor, l0bTensor, para);\
    GlobalMockObject::verify();\
}

#define MMAD_TEST(D, S, M, N, K, KA, CS)    MMAD_TEST_BASE(D, S, M, N, K, KA, CS, mad)
#define MMAD_MX_TEST(D, S, M, N, K, KA, CS) MMAD_TEST_BASE(D, S, M, N, K, KA, CS, mad_mx)


MMAD_TEST(float, half, 16, 16, 16, 0, 0);
MMAD_TEST(float, float, 16, 16, 16, 0, 0);
MMAD_TEST(float, bfloat16_t, 16, 16, 16, 0, 0);

// MMAD_MX_TEST(float, fp4x2_e2m1_t, 32, 32, 32, 0, 0);
// MMAD_MX_TEST(float, fp4x2_e1m2_t, 32, 32, 32, 0, 0);
// MMAD_MX_TEST(float, fp8_e4m3fn_t, 32, 32, 32, 0, 0);
// MMAD_MX_TEST(float, fp8_e5m2_t, 32, 32, 32, 0, 0);