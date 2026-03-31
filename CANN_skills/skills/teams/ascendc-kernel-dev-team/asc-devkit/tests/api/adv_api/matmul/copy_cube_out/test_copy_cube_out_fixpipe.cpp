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
 * \file test_copy_cube_out_fixpipe.cpp
 * \brief
 */
 
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_out/quant/quant_processor_fixpipe.h"

using namespace std;
using namespace AscendC;

namespace {
const uint64_t NUM_SIXTEEN = 16;

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG, typename = void>
class CustomBatchLoop
{
public:
    __aicore__ inline int32_t GetBatchOutCacheNum() const
    {
        return 1;
    }
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeOut = Impl::Detail::CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::SINGLE_DST_MODE>;
    using MatmulSubBlockInfo = Impl::Detail::MatmulSubBlockInfo<IMPL, MM_CFG>;
    using BatchLoop = CustomBatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class SplitMMatmulPolicy : public Impl::Detail::SplitMMatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeOut = Impl::Detail::CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::DUAL_DST_SPLIT_M>;
    using MatmulSubBlockInfo = Impl::Detail::MatmulSubBlockInfo<IMPL, MM_CFG>;
    using BatchLoop = CustomBatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class SplitNMatmulPolicy : public Impl::Detail::SplitNMatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CopyCubeOut = Impl::Detail::CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::DUAL_DST_SPLIT_N>;
    using MatmulSubBlockInfo = Impl::Detail::MatmulSubBlockInfo<IMPL, MM_CFG>;
    using BatchLoop = CustomBatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, 
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeOutBuffer) 
, MATMUL_IMPORT_MODULE(CopyCubeOut)
, MATMUL_IMPORT_MODULE(BatchLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulQuantProcessor)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUserDefineInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulSubBlockInfo)
{
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING(BatchLoop);
    MATMUL_ALLOW_USING_PRIVATE(MatmulQuantProcessor);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUserDefineInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulSubBlockInfo);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using CallBack = MM_CB;
    using AscendC::Impl::Detail::MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG>::SetQuantVector;

    CubeOutBuffer::Init;
    CubeOutBuffer::AllocTensor;
    CubeOutBuffer::GetTensor;
    CubeOutBuffer::EnQue;
    CubeOutBuffer::DeQue;
    CubeOutBuffer::FreeTensor;
    CopyCubeOut::Copy;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulQuantProcessor);

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

    void InitQuant(int32_t size) {
        MATMUL_MODULE(MatmulQuantProcessor)->Init(size);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
};

__aicore__ constexpr MatmulConfig GetIntraBlockPartSumCFG()
{
    auto mmCfg = CFG_NORM;
    mmCfg.intraBlockPartSum = true;
    mmCfg.isEnableChannelSplit = true;

    return mmCfg;
}

__aicore__ constexpr MatmulConfig GetIsBasicCFG()
{
    auto mmCfg = CFG_NORM;
    mmCfg.singleCoreM = mmCfg.basicM;
    mmCfg.singleCoreN = mmCfg.basicN;
    mmCfg.singleCoreK = mmCfg.basicK;
    mmCfg.enableSetTail = true;
    mmCfg.bmmOutMode = BatchOutMode::MULTI_BATCH;

    return mmCfg;
}
__aicore__ TCubeTiling GetTilingTest(TCubeTiling tiling)
{
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
    return tiling;
}
class TestCopyCubeOutFixpipe : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true, LayoutMode::NONE, true>;
    using A_TYPE_Quant = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, fp8_e4m3fn_t, true, LayoutMode::NONE, true>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true, LayoutMode::NONE, true>;
    using B_TYPE_Quant = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, fp8_e4m3fn_t, true, LayoutMode::NONE, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true, LayoutMode::NONE, true>;
    using C_TYPE_Quant = MatmulType<AscendC::TPosition::VECIN, CubeFormat::NZ, bfloat16_t, true, LayoutMode::BSNGD, true>;
    using C_TYPE_UB = MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, float, true, LayoutMode::BSNGD, true>;
    using C_TYPE_SBNGD = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float, true, LayoutMode::SBNGD, true>;
    using C_TYPE_BSNGD = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float, true, LayoutMode::BSNGD, true>;
    using C_TYPE_NZ_SBNGD = MatmulType<AscendC::TPosition::VECIN, CubeFormat::NZ, float, true, LayoutMode::SBNGD, true>;
    using C_TYPE_NZ_BSNGD = MatmulType<AscendC::TPosition::VECIN, CubeFormat::NZ, float, true, LayoutMode::BSNGD, true>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    static void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *DataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr){};
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<TestCopyCubeOutFixpipe::DataCopyOut, nullptr, nullptr>, CustomMatmulPolicy> mm1;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_NZ_SBNGD, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<TestCopyCubeOutFixpipe::DataCopyOut, nullptr, nullptr>, CustomMatmulPolicy> mm2;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_SBNGD, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, SplitMMatmulPolicy> mm3;
    MatmulImpl<A_TYPE_Quant, B_TYPE_Quant, C_TYPE_Quant, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, SplitMMatmulPolicy> mm4;
    static constexpr MatmulConfig IB_CFG = GetIsBasicCFG();
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_BSNGD, BIAS_TYPE, IB_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, SplitNMatmulPolicy> mm5;
    static constexpr MatmulConfig IBPS_CFG = GetIntraBlockPartSumCFG();
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, IBPS_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm6;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_UB, BIAS_TYPE, IBPS_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm7;
};

TEST_F(TestCopyCubeOutFixpipe, CopyOutImplCB){
    TCubeTiling tiling;
    auto tilingResult = GetTilingTest(tiling);
    mm1.InitVar(tilingResult);
    mm1.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm1.Init(mm1.GetBufferSize());
    mm1.AllocTensor();
    auto co1Local1 = mm1.GetTensor();
    mm1.EnQue(co1Local1);
    mm1.DeQue();
    const int data_size = NUM_SIXTEEN * NUM_SIXTEEN;
    uint8_t aGM[data_size * sizeof(C_TYPE::T)] = {0};
    GlobalTensor<C_TYPE::T> dst1;
    dst1.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE::T*>(aGM), data_size);
    mm1.template Copy<false>(dst1, co1Local1, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm1.template Copy<true>(dst1, co1Local1, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm1.FreeTensor(co1Local1);

    mm2.InitVar(tilingResult);
    mm2.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm2.Init(mm2.GetBufferSize());
    mm2.AllocTensor();
    auto co1Local2 = mm2.GetTensor();
    mm2.EnQue(co1Local2);
    mm2.DeQue();
    uint8_t bGM[data_size * sizeof(C_TYPE::T)] = {0};
    GlobalTensor<C_TYPE::T> dst2;
    dst2.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE::T*>(bGM), data_size);
    mm2.template Copy<false>(dst2, co1Local2, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm2.template Copy<true>(dst2, co1Local2, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm2.FreeTensor(co1Local2);
}

TEST_F(TestCopyCubeOutFixpipe, CopyCallBack){
    TCubeTiling tiling;
    auto tilingResult = GetTilingTest(tiling);
    mm3.InitVar(tilingResult);
    mm3.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm3.Init(mm3.GetBufferSize());
    mm3.AllocTensor();
    auto co1Local1 = mm3.GetTensor();
    mm3.EnQue(co1Local1);
    mm3.DeQue();
    const int data_size = NUM_SIXTEEN * NUM_SIXTEEN;
    uint8_t aGM[data_size * sizeof(C_TYPE_SBNGD::T)] = {0};
    GlobalTensor<C_TYPE_SBNGD::T> dst1;
    dst1.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE_SBNGD::T*>(aGM), data_size);
    mm3.template Copy<false>(dst1, co1Local1, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm3.FreeTensor(co1Local1);

    mm4.InitVar(tilingResult);
    mm4.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm4.Init(mm4.GetBufferSize());
    mm4.InitQuant(NUM_SIXTEEN);
    mm4.AllocTensor();
    auto co1Local2 = mm4.GetTensor();
    mm4.EnQue(co1Local2);
    mm4.DeQue();
    uint8_t quantGM[data_size * sizeof(uint64_t)] = {0};
    GlobalTensor<uint64_t> gTensor;
    gTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(quantGM), data_size);
    mm4.SetQuantVector(gTensor);
    uint8_t bGM[data_size * sizeof(C_TYPE_Quant::T)] = {0};
    GlobalTensor<C_TYPE_Quant::T> dst2;
    dst2.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE_Quant::T*>(bGM), data_size);
    mm4.template Copy<false>(dst2, co1Local2, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm4.FreeTensor(co1Local2);

    mm5.InitVar(tilingResult);
    mm5.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm5.Init(mm5.GetBufferSize());
    mm5.AllocTensor();
    auto co1Local3 = mm5.GetTensor();
    mm5.EnQue(co1Local3);
    mm5.DeQue();
    uint8_t cGM[data_size * sizeof(C_TYPE_NZ_BSNGD::T)] = {0};
    GlobalTensor<C_TYPE_NZ_BSNGD::T> dst3;
    dst3.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE_NZ_BSNGD::T*>(cGM), data_size);
    mm5.template Copy<false>(dst3, co1Local3, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm5.FreeTensor(co1Local3);
}

TEST_F(TestCopyCubeOutFixpipe, Copy_Local_To_Local){
    TCubeTiling tiling;
    auto tilingResult = GetTilingTest(tiling);
    mm3.InitVar(tilingResult);
    mm3.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm3.Init(mm3.GetBufferSize());
    mm3.AllocTensor();
    auto co1Local1 = mm3.GetTensor();
    mm3.EnQue(co1Local1);
    mm3.DeQue();
    const int data_size = NUM_SIXTEEN * NUM_SIXTEEN;
    uint64_t localAddr = 0;
    LocalTensor<C_TYPE_SBNGD::T> dst1(TPosition::A1, localAddr, data_size);
    mm3.template Copy<false>(dst1, co1Local1, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm3.FreeTensor(co1Local1);

    mm4.InitVar(tilingResult);
    mm4.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm4.Init(mm4.GetBufferSize());
    mm4.InitQuant(NUM_SIXTEEN);
    mm4.AllocTensor();
    auto co1Local2 = mm4.GetTensor();
    mm4.EnQue(co1Local2);
    mm4.DeQue();
    uint8_t quantGM[data_size * sizeof(uint64_t)] = {0};
    GlobalTensor<uint64_t> gTensor;
    gTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(quantGM), data_size);
    mm4.SetQuantVector(gTensor);
    LocalTensor<C_TYPE_Quant::T> dst2(TPosition::A1, localAddr, data_size);
    mm4.template Copy<false>(dst2, co1Local2, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm4.FreeTensor(co1Local2);

    mm5.InitVar(tilingResult);
    mm5.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm5.Init(mm5.GetBufferSize());
    mm5.AllocTensor();
    auto co1Local3 = mm5.GetTensor();
    mm5.EnQue(co1Local3);
    mm5.DeQue();
    LocalTensor<C_TYPE_NZ_BSNGD::T> dst3(TPosition::A1, localAddr, data_size);
    mm5.template Copy<false>(dst3, co1Local3, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm5.FreeTensor(co1Local3);
}

TEST_F(TestCopyCubeOutFixpipe, CopyIntraBlockPartSum){
    TCubeTiling tiling;
    auto tilingResult = GetTilingTest(tiling);
    mm6.InitVar(tilingResult);
    mm6.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm6.Init(mm6.GetBufferSize());
    mm6.AllocTensor();
    auto co1Local1 = mm6.GetTensor();
    mm6.EnQue(co1Local1);
    mm6.DeQue();
    const int data_size = NUM_SIXTEEN * NUM_SIXTEEN;
    uint8_t aGM[data_size * sizeof(C_TYPE::T)] = {0};
    GlobalTensor<C_TYPE::T> dst1;
    dst1.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE::T*>(aGM), data_size);
    mm6.template Copy<false>(dst1, co1Local1, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm6.FreeTensor(co1Local1);

    mm7.InitVar(tilingResult);
    mm7.SetRuntimeParams(NUM_SIXTEEN, NUM_SIXTEEN);
    mm7.Init(mm7.GetBufferSize());
    mm7.AllocTensor();
    auto co1Local2 = mm7.GetTensor();
    mm7.EnQue(co1Local2);
    mm7.DeQue();
    uint8_t bGM[data_size * sizeof(C_TYPE_UB::T)] = {0};
    GlobalTensor<C_TYPE_UB::T> dst2;
    dst2.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE_UB::T*>(bGM), data_size);
    mm7.template Copy<false>(dst2, co1Local2, 0, 0, NUM_SIXTEEN, NUM_SIXTEEN, 1, 1);
    mm7.FreeTensor(co1Local2);
}
} //namespace
