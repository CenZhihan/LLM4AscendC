/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file test_matmul_load_to_l0a_gemv_mx.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_gemv.h"

using namespace std;
using namespace AscendC;

namespace {

constexpr int32_t MX_K_FACTOR = 32;

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using LoadToA2 = Impl::Detail::LoadToL0A<IMPL, A_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(LoadToA2) {
    MATMUL_ALLOW_USING(LoadToA2);

public:
    MatmulImpl() = default;
    TPipe pipe;

};

class TestMxMatmulLoadToL0aGemv : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_FP8_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using A_FP4_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using B_FP8_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;
    using B_FP4_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e1m2_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;

    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_FP8_TYPE, B_FP8_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void, CustomMatmulPolicy> mm1;
    MatmulImpl<A_FP4_TYPE, B_FP4_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void, CustomMatmulPolicy> mm2;
};

TEST_F(TestMxMatmulLoadToL0aGemv, MxMatmulLoadL1ToL0aGemvFp8Case) {
    // input: M: 1, K: 1024
    const uint32_t M = 1;
    const uint32_t K = 1024;

    TBuf<TPosition::A2> l0aBuf;
    mm1.pipe.InitBuffer(l0aBuf, M * K);
    auto l0a = l0aBuf.Get<mx_fp8_e5m2_t>();
    TBuf<TPosition::TSCM> l1aBuf;
    mm1.pipe.InitBuffer(l1aBuf, M * K);
    auto l1a = l1aBuf.Get<fp8_e5m2_t>();
    TBuf<TPosition::TSCM> l1aMxBuf;
    mm1.pipe.InitBuffer(l1aMxBuf, M * K / MX_K_FACTOR);
    auto l1aMx = l1aMxBuf.Get<fp8_e8m0_t>();

    mm1.Load(l0a, l1a, 0, 0, M, K, 0, 0, false, l1aMx, M * K / MX_K_FACTOR, 0 , 0);
}

TEST_F(TestMxMatmulLoadToL0aGemv, MxMatmulLoadL1ToL0aGemvFp4Case) {
    // input: M: 1, K: 1024
    const uint32_t M = 1;
    const uint32_t K = 1024;

    TBuf<TPosition::A2> l0aBuf;
    mm2.pipe.InitBuffer(l0aBuf, M * K);
    auto l0a = l0aBuf.Get<fp4x2_e2m1_t>();
    TBuf<TPosition::TSCM> l1aBuf;
    mm2.pipe.InitBuffer(l1aBuf, M * K);
    auto l1a = l1aBuf.Get<fp4x2_e2m1_t>();
    TBuf<TPosition::TSCM> l1aMxBuf;
    mm2.pipe.InitBuffer(l1aMxBuf, M * K / MX_K_FACTOR);
    auto l1aMx = l1aMxBuf.Get<fp8_e8m0_t>();

    mm2.Load(l0a, l1a, 0, 0, M, K, 0, 0, false, l1aMx, M * K / MX_K_FACTOR, 0 , 0);
}
} //namespace