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
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a.h"

using namespace std;
using namespace AscendC;


namespace {

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
    MatmulImpl() {};
    TPipe pipe;

};
}

class test_matmul_load_to_l0a : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using L0cT = float;

    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, L0cT>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void, CustomMatmulPolicy> mm;
};

TEST_F(test_matmul_load_to_l0a, case0) {
    // input: M : 16, K : 16
    TBuf<TPosition::A2> l0aBuf;
    mm.pipe.InitBuffer(l0aBuf, 65536);
    auto l0a = l0aBuf.Get<half>();
    TBuf<TPosition::TSCM> l1aBuf;
    mm.pipe.InitBuffer(l1aBuf, 65536);
    auto l1a = l1aBuf.Get<half>();
    mm.SetScalar(0);
    mm.Prepare(false, 16, 16);
    mm.Load(l0a, l1a, 16, 16, 16, 16, 0, 0, false);
}
