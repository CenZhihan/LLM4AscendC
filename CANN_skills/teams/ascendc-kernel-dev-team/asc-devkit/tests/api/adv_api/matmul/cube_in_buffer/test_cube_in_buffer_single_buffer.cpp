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
#include "impl/adv_api/detail/matmul/resource/cube_in_buffer/cube_in_buffer.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"

using namespace std;
using namespace AscendC;


namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CubeInBufferA = Impl::Detail::CubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = Impl::Detail::CubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeInBufferB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeInBufferB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

public:
    using CubeInBufferB::Init;
    using CubeInBufferB::Destroy;
    using CubeInBufferB::AllocTensor;
    using CubeInBufferB::FreeTensor;
    using CubeInBufferB::Hit;
    using CubeInBufferB::GetBuffer;
    using CubeInBufferB::Reset;
    using CubeInBufferB::EnQue;
    using CubeInBufferB::DeQue;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    
    MatmulImpl() {
        InitVar();
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetInitParams(int32_t stepN, int32_t stepKb, int32_t baseN, int32_t baseK) {
        tiling.stepN = stepN;
        tiling.stepKb = stepKb;
        tiling.baseN = baseN;
        tiling.baseK = baseK;
        tiling.iterateOrder = 0;
    }

    void SetRuntimeParams(int32_t baseUseN, int32_t baseUseK) {
        var.baseUseN_ = baseUseN;
        var.baseUseK_ = baseUseK;
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

constexpr MatmulConfig MM_CFG_CUSTOM { true, false, false, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0,  0, 0,
    false, false, false, false, false, true, BatchMode::NONE, true, true, true, true, true, true, true,
    IterateMode::ITERATE_MODE_DEFAULT, false, true, false, true, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
    false, true};
class test_cube_in_buffer_single_buffer : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE_BMM = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE_BMM, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_CUSTOM, void, CustomMatmulPolicy> mm;
};

TEST_F(test_cube_in_buffer_single_buffer, DISABLED_get_iter_index) {
    mm.SetInitParams(2, 2, 32, 32);
    int32_t mIter = 2;
    int32_t kIter = 3;
    mm.Init(1024, 4);
}

TEST_F(test_cube_in_buffer_single_buffer, DISABLED_all_interface_normal) {
    mm.SetInitParams(2, 2, 32, 32);
    int32_t mIter = 2;
    int32_t kIter = 2;
    int32_t hitCnt = 0;
    mm.Init(1024, 4);
    LocalTensor<half> fakeTensor;
    for (int32_t m = 0; m < mIter; m++) {
        for (int32_t k = 0; k < kIter; k++) {
            int32_t iterIndex = 0;
            if (mm.Hit(iterIndex)) {
                fakeTensor = mm.GetBuffer(iterIndex);
                hitCnt++;
            } else {
                fakeTensor = mm.AllocTensor(iterIndex);
                mm.EnQue(fakeTensor);
                mm.DeQue();
            }
            mm.FreeTensor(iterIndex, fakeTensor);
        }
        mm.Reset();
    }
    mm.Destroy();
    ASSERT_EQ(hitCnt, 0);
}
