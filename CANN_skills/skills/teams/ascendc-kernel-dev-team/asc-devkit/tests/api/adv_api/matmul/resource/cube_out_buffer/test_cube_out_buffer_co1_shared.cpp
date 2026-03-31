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
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "impl/adv_api/detail/matmul/feature_trait/matmul_feature_trait.h"
 
using namespace std;

namespace AscendC {
template <typename T>
const LocalTensor<T> EMPTY_TENSOR;
constexpr int NUM_SIXTEEN = 16;

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {};
 
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE(CopyCubeOut)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
 
    using SrcT = typename A_TYPE::T;
 
public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
 
    using CallBack = MM_CB;
 
    CubeOutBuffer::Init;
    CubeOutBuffer::AllocTensor;
    CubeOutBuffer::GetTensor;
    CubeOutBuffer::EnQue;
    CubeOutBuffer::DeQue;
    CubeOutBuffer::FreeTensor;
    CopyCubeOut::Copy;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
 
    MatmulImpl() {}
 
    VAR_PARAMS& GetVar() {
        return var;
    }
 
    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }
 
    void SetRuntimeParams(int32_t m, int32_t n) {
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgShape(m, n, n, n, 0);
    }
 
    uint32_t GetBufferSize()
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        return tiling.GetBaseM() * tiling.GetBaseN();
    }
 
private:
    TPipe pipe;
    VAR_PARAMS var;
};
 
class TestCopyCubeOut : public testing::Test {
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using C_TYPE_NZ = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using A_T = A_TYPE::T;
 
protected:
    void SetUp() {}
    void TearDown() {}
};
 
__aicore__ constexpr MatmulConfig GetCo1Config()
{
    auto mmCfg = CFG_NORM;
    mmCfg.isCO1Shared = true;
    
    return mmCfg;
}
 
TEST_F(TestCopyCubeOut, Co1_shared) {
    AscendC::TQue<AscendC::QuePosition::CO1, 1, &AscendC::gCO1Config> qCO1;
    AscendC::gCO1Que = &qCO1;
    static constexpr MatmulConfig co1Config = GetCo1Config();
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, co1Config, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
 
    TCubeTiling tiling;
    tiling.M = NUM_SIXTEEN;
    tiling.N = NUM_SIXTEEN;
    tiling.Ka = NUM_SIXTEEN;
    tiling.Kb = NUM_SIXTEEN;
    tiling.singleCoreM = NUM_SIXTEEN;
    tiling.singleCoreN = NUM_SIXTEEN;
    tiling.singleCoreK = NUM_SIXTEEN;
    tiling.baseM = NUM_SIXTEEN;
    tiling.baseN = NUM_SIXTEEN;
    tiling.baseK = NUM_SIXTEEN;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = false;
    tiling.iterateOrder = 0;
    mm.InitVar(tiling);
 
    mm.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
 
    mm.Init(mm.GetBufferSize());
    mm.AllocTensor();
    auto co1Local = mm.GetTensor();
    mm.EnQue(co1Local);
    mm.DeQue();
}
}
