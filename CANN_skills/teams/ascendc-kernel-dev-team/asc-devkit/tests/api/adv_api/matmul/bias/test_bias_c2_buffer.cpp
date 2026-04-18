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
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/resource/bias_buffer/c2_buffer/c2_buffer.h"

using namespace std;
using namespace AscendC;


namespace {
__aicore__ inline constexpr MatmulConfig GetMmConfig()
{
    auto cfg = GetMDLConfig();
    cfg.enableSetBias = false;
    return cfg;
}
constexpr MatmulConfig CFG_NO_BIAS = GetMmConfig();

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using C2Buffer = Impl::Detail::C2Buffer<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(C2Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(C2Buffer);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    using VAR_PARAMS = typename Impl::Detail::MatmulParams<
        A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;

public:
    using C2Buffer::Init;
    using C2Buffer::Allocate;
    using C2Buffer::EnQue;
    using C2Buffer::DeQue;
    using C2Buffer::Free;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    MatmulImpl() {
        SetGCoreType(1);
        InitVar();
    }

    ~MatmulImpl() {
        SetGCoreType(0);
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void DestroyTPipe() {
        pipe.Destroy();
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestMatmulC2Buffer : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

    void RunCase(auto& mm) {
        mm.Init();
        auto bias = mm.Allocate();
        mm.EnQue();
        mm.DeQue();
        mm.Free();
        mm.DestroyTPipe();
    }

private:
    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type; 

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void, CustomMatmulPolicy> mm1_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NO_BIAS, void, CustomMatmulPolicy> mm2_;
    // MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void, CustomMatmulPolicy> mm3_;
    // MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NO_BIAS, void, CustomMatmulPolicy> mm4_;
};

TEST_F(TestMatmulC2Buffer, c2_buffer_with_bias_mdl) {
    RunCase(mm1_);
}

TEST_F(TestMatmulC2Buffer, c2_buffer_with_no_bias_mdl) {
    RunCase(mm2_);
}
