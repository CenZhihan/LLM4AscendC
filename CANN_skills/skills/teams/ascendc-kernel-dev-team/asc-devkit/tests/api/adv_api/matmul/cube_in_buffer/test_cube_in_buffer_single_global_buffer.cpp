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
: MATMUL_IMPORT_MODULE(CubeInBufferA)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeInBufferA);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

public:
    using CubeInBufferA::Init;
    using CubeInBufferA::Destroy;
    using CubeInBufferA::AllocTensor;
    using CubeInBufferA::FreeTensor;
    using CubeInBufferA::Hit;
    using CubeInBufferA::GetBuffer;
    using CubeInBufferA::Reset;
    using CubeInBufferA::EnQue;
    using CubeInBufferA::DeQue;
    using CubeInBufferA::SetOrgTensor;
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

    void SetInitParams(int32_t stepM, int32_t stepKa, int32_t baseM, int32_t baseK) {
        tiling.stepM = stepM;
        tiling.stepKa = stepKa;
        tiling.baseM = baseM;
        tiling.baseK = baseK;
        tiling.iterateOrder = 0;
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class test_cube_in_buffer_single_global_buffer : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE_IBSHARE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, true>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE_IBSHARE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM, void, CustomMatmulPolicy> mm;
    Impl::Detail::GlobalCache gCache;
};

TEST_F(test_cube_in_buffer_single_global_buffer, DISABLED_cube_in_buffer_global_get_iter_index) {
    gCache.Init();
    int32_t mIter = 2;
    int32_t kIter = 3;
    mm.SetInitParams(2, 2, 32, 32);
    mm.Init(1024, 4);
}

TEST_F(test_cube_in_buffer_single_global_buffer, DISABLED_all_interface_hit_none) {
    gCache.Init();
    GlobalTensor<half> fakeInput;
    int32_t mIter = 2;
    int32_t kIter = 2;
    int32_t nIter = 2;
    int32_t hitCnt = 0;
    mm.SetInitParams(2, 2, 32, 32);
    mm.Init(1024, 4);
    mm.SetOrgTensor(fakeInput);
    LocalTensor<half> fakeTensor;
    for (int32_t m = 0; m < mIter; m++) {
        for (int32_t n = 0; n < nIter; n++) {
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
            }
        }
        mm.Reset();
    }
    mm.Destroy();
    ASSERT_EQ(hitCnt, 0);
}

TEST_F(test_cube_in_buffer_single_global_buffer, DISABLED_all_interface_normal_hit) {
    gCache.Init();
    GlobalTensor<half> fakeInput;
    int32_t mIter = 2;
    int32_t kIter = 2;
    int32_t nIter = 2;
    int32_t hitCnt = 0;
    mm.SetInitParams(2, 2, 32, 32);
    mm.Init(1024, 4);
    mm.SetOrgTensor(fakeInput);
    LocalTensor<half> fakeTensor;
    for (int32_t m = 0; m < mIter; m++) {
        for (int32_t n = 0; n < nIter; n++) {
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
            }
        }
        mm.Reset();
    }
    mm.Destroy();
    ASSERT_EQ(hitCnt, 6);
}

TEST_F(test_cube_in_buffer_single_global_buffer, DISABLED_all_interface_reduce_cache) {
    gCache.Init();
    int32_t mIter = 2;
    int32_t kIter = 2;
    int32_t nIter = 2;
    int32_t hitCnt = 0;
    mm.SetInitParams(2, 2, 32, 32);
    mm.Init(1024, 4);
    GlobalTensor<half> fakeInput;
    mm.SetOrgTensor(fakeInput);
    LocalTensor<half> fakeTensor;
    for (int32_t m = 0; m < mIter; m++) {
        for (int32_t n = 0; n < nIter; n++) {
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
            }
        }
        mm.Reset();
    }
    mm.Destroy();
    ASSERT_EQ(hitCnt, 6);
}
