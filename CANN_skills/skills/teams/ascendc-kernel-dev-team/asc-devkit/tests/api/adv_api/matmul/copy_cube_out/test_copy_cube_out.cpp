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

using namespace std;
using namespace AscendC;

namespace {
template <typename T>
const LocalTensor<T> EMPTY_TENSOR;

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{

};

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
}

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

private:
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_NZ, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;
};

TEST_F(TestCopyCubeOut, Copy_NZ_From_ND) {

    TCubeTiling tiling;
    tiling.M = 16;
    tiling.N = 16;
    tiling.Ka = 16;
    tiling.Kb = 16;
    tiling.singleCoreM = 16;
    tiling.singleCoreN = 16;
    tiling.singleCoreK = 16;
    tiling.baseM = 16;
    tiling.baseN = 16;
    tiling.baseK = 16;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = false;
    tiling.iterateOrder = 0;
    mm.InitVar(tiling);

    mm.SetRuntimeParams(16, 16);

    mm.Init(mm.GetBufferSize());
    mm.AllocTensor();
    auto co1Local = mm.GetTensor();
    mm.EnQue(co1Local);
    mm.DeQue();

    const int data_size = 16 * 16;
    uint8_t aGM[data_size * sizeof(A_T)] = {0};
    GlobalTensor<A_T> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), data_size);
    mm.template Copy<false>(dst, co1Local, 0, 0, 16, 16, 1, 1);

    mm.FreeTensor(co1Local);
}

TEST_F(TestCopyCubeOut, Copy_NZ_From_NZ) {

    TCubeTiling tiling;
    tiling.M = 16;
    tiling.N = 16;
    tiling.Ka = 16;
    tiling.Kb = 16;
    tiling.singleCoreM = 16;
    tiling.singleCoreN = 16;
    tiling.singleCoreK = 16;
    tiling.baseM = 16;
    tiling.baseN = 16;
    tiling.baseK = 16;
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = false;
    tiling.iterateOrder = 0;
    mm1.InitVar(tiling);

    mm1.SetRuntimeParams(16, 16);

    mm1.Init(mm1.GetBufferSize());
    mm1.AllocTensor();
    auto co1Local = mm1.GetTensor();
    mm1.EnQue(co1Local);
    mm1.DeQue();

    const int data_size = 16 * 16;
    uint8_t aGM[data_size * sizeof(A_T)] = {0};
    GlobalTensor<A_T> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), data_size);
    mm1.template Copy<false>(dst, co1Local, 0, 0, 16, 16, 1, 1);

    mm1.FreeTensor(co1Local);
}
