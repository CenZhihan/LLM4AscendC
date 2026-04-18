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
#include "impl/adv_api/detail/matmul/stage/split/load_to_l0b/load_to_l0b.h"
#include "impl/adv_api/detail/matmul/stage/compute/mmad_compute.h"
#include "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0.h"

using namespace std;
using namespace AscendC;


namespace {

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using LoadToA2 = Impl::Detail::LoadToL0A<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = Impl::Detail::LoadToL0B<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using MmadCompute = Impl::Detail::MmadCompute<IMPL, float, A_TYPE, typename B_TYPE::T, MM_CFG>;
    using TBufPoolL0 = Impl::Detail::TBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(TBufPoolL0)
, MATMUL_IMPORT_MODULE(LoadToA2)
, MATMUL_IMPORT_MODULE(LoadToB2)
, MATMUL_IMPORT_MODULE(MmadCompute)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(TBufPoolL0);
    MATMUL_ALLOW_USING(LoadToA2);
    MATMUL_ALLOW_USING(LoadToB2);
    MATMUL_ALLOW_USING(MmadCompute);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    using A_T = typename A_TYPE::T;

public:
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(LoadToA2);
    MATMUL_USE_MODULE(LoadToB2);
    MatmulImpl() {
        pipe = *GetTPipePtr();
    }

    TPipe pipe;

    void InitA(bool isATranspose, uint16_t aL1K, uint16_t aL1M)
    {
        MATMUL_MODULE(LoadToA2)->Prepare(isATranspose, aL1K, aL1M);
    }

    void LoadA(LocalTensor<A_T> &l0A, const LocalTensor<A_T> &l1A,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose)
     {
        MATMUL_MODULE(LoadToA2)->Load(l0A, l1A, aL1M, aL1K, madM, madK, aL1MOffset, aL1KOffset, isATranspose);
     }

     void InitB(bool isBTranspose, uint16_t bL1K)
     {
        MATMUL_MODULE(LoadToB2)->Prepare(isBTranspose, bL1K);
     }

     void LoadB(LocalTensor<A_T> &l0B, const LocalTensor<A_T> &l1B,
     uint16_t bL1N, uint16_t bL1K, uint16_t madN, uint16_t madK, uint16_t bL1NOffset, uint16_t bL1KOffset,
     bool isBTranspose)
     {
        MATMUL_MODULE(LoadToB2)->Load(l0B, l1B, bL1N, bL1K, madN, madK, bL1NOffset, bL1KOffset, isBTranspose);
     }
};
}

class test_matmul_tbufpool_l0 : public testing::Test {
protected:
    void SetUp() {
        SetGCoreType(1);
    }
    void TearDown() {
        SetGCoreType(0);
    }
};

TEST_F(test_matmul_tbufpool_l0, case0) {
    // input: M : 16, K : 16, N : 16, isATrans : false, isBTrans : false
    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void, CustomMatmulPolicy> mm;
    TCubeTiling tiling;
    mm.SetTiling(&tiling);
    mm.InitA(false, 16, 16);
    mm.InitB(false, 16);
    mm.Init();
    mm.SetDBFlag(false);
    auto& bufferPool = mm.Allocate();
    auto l0a = bufferPool.template GetBuffer<TPosition::A2, half>();
    auto l0b = bufferPool.template GetBuffer<TPosition::B2, half>();
    TBuf<TPosition::TSCM> l1aBuf;
    mm.pipe.InitBuffer(l1aBuf, 65536);
    auto l1a = l1aBuf.Get<half>();
    TBuf<TPosition::TSCM> l1bBuf;
    mm.pipe.InitBuffer(l1bBuf, 65536);
    auto l1b = l1bBuf.Get<half>();
    mm.LoadA(l0a, l1a, 16, 16, 16, 16, 0, 0, false);
    mm.LoadB(l0b, l1b, 16, 16, 16, 16, 0, 0, false);
    bufferPool.EnQue();
    bufferPool.DeQue();
    TBuf<TPosition::CO1> co1BUf;
    mm.pipe.InitBuffer(co1BUf, 65536);
    auto l0c = co1BUf.Get<float>();
    mm.Compute(l0c, l0a, l0b, 16, 16, 16, false, false);
    bufferPool.Free();
}

TEST_F(test_matmul_tbufpool_l0, case1) {
    // input: M : 16, K : 32, N : 64, isATrans : true, isBTrans : true
    using A_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using B_TYPE = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, half>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void, CustomMatmulPolicy> mm;
    TCubeTiling tiling;
    mm.SetTiling(&tiling);
    mm.InitA(true, 32, 16);
    mm.InitB(true, 32);
    mm.Init();
    mm.SetDBFlag(true);
    auto& bufferPool = mm.Allocate();
    auto l0a = bufferPool.template GetBuffer<TPosition::A2, half>();
    auto l0b = bufferPool.template GetBuffer<TPosition::B2, half>();
    TBuf<TPosition::TSCM> l1aBuf;
    mm.pipe.InitBuffer(l1aBuf, 65536);
    auto l1a = l1aBuf.Get<half>();
    TBuf<TPosition::TSCM> l1bBuf;
    mm.pipe.InitBuffer(l1bBuf, 65536);
    auto l1b = l1bBuf.Get<half>();
    mm.LoadA(l0a, l1a, 16, 32, 16, 32, 0, 0, true);
    mm.LoadB(l0b, l1b, 64, 32, 64, 32, 0, 0, true);
    bufferPool.EnQue();
    bufferPool.DeQue();
    TBuf<TPosition::CO1> co1BUf;
    mm.pipe.InitBuffer(co1BUf, 65536);
    auto l0c = co1BUf.Get<float>();
    mm.Compute(l0c, l0a, l0b, 16, 32, 64, true, true);
    bufferPool.Free();
}
