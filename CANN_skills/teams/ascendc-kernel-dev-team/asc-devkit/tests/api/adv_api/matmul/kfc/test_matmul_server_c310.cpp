/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#define ENABLE_CV_COMM_VIA_SSBUF true // kfc 310 mode

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl.h"
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl_3510.h"
#include "kfc_fake_modules.h"
#include "../copy_cube_in/base_tiling_struct.h"

// UT not support multi thread, only aic or aiv thread, func that contain aic and aiv can not simulated
using namespace std;

namespace AscendC {
constexpr uint32_t NUM_FIFTYSIX = 56;
template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using Scheduler = CustomMatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using C1Buffer = Impl::Detail::C1Buffer<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
};

template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class BatchCustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using BatchScheduler = CustomBatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulWithScalePolicy :
    public Impl::Detail::MatmulWithScalePolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using Scheduler = CustomMatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
};

class TestMatmulServerC310 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using A_TYPE_IBSHARE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, true>;
    using A_TYPE_NORMAL_SCALAR = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;

    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using B_TYPE_IBSHARE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, true>;
    using B_TYPE_NORMAL_SCALAR = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NORMAL>;

    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using C_TYPE_UB = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
    using C_TYPE_NORMAL_SCALAR = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;

    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    using A_MX_TSCM_TYPE = MatmulTypeWithScale<AscendC::TPosition::TSCM, AscendC::TPosition::TSCM, CubeFormat::ND,
        fp8_e5m2_t, true>;
    using B_MX_TSCM_TYPE = MatmulTypeWithScale<AscendC::TPosition::TSCM, AscendC::TPosition::TSCM, CubeFormat::ND,
        fp8_e5m2_t, true>;

    using A_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD>;
    using B_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BSNGD>;
    using C_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BSNGD>;

    using A_TYPE_BNGS1S2 = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BNGS1S2>;
    using B_TYPE_BNGS1S2 = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BNGS1S2>;
    using C_TYPE_BNGS1S2 = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
};

__aicore__ constexpr MatmulConfig GetBmmNBatchConfig()
{
    MatmulBatchParams bmmParams = {true, BatchMode::BATCH_LESS_THAN_L1, true};
    auto bmmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(bmmParams);
    return bmmCfg;
}

__aicore__ constexpr MatmulConfig GetN_BATCH_Config()
{
    auto mmCfg = CFG_NORM;
    mmCfg.enableSetBias = false;
    mmCfg.enableSetTail = true;
    mmCfg.enableSetOrgShape = false;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_N_BATCH;

    return mmCfg;
}

void setTiling() {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);

    MSG_POS TilingInfo *tilingSSbuf = reinterpret_cast<MSG_POS TilingInfo *>(GetTilingAddr(GetSubBlockIdxImpl()));
    tilingSSbuf->valid = 1;
    auto tempTilingSSbuf = reinterpret_cast<MSG_POS uint64_t*>(&(tilingSSbuf->tCubeTiling));
    auto tempTiling = reinterpret_cast<uint64_t*>(&tiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
        *tempTilingSSbuf = *tempTiling;
    }
}

void setBatchTiling() {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParamsBatch tilingParams = {1, 128, 64, 256, 128, 64, 256, 128, 64, 128, 2, 1, 1, 1, 2, 2, 0, 0, 2, 2, 2, 2, 128, 2, 1, 256, 2, 64, 2, 1, 256, 2, 128, 2, 1, 64};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);

    MSG_POS TilingInfo *tilingSSbuf = reinterpret_cast<MSG_POS TilingInfo *>(GetTilingAddr(GetSubBlockIdxImpl()));
    tilingSSbuf->valid = 1;
    auto tempTilingSSbuf = reinterpret_cast<MSG_POS uint64_t*>(&(tilingSSbuf->tCubeTiling));
    auto tempTiling = reinterpret_cast<uint64_t*>(&tiling);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
        *tempTilingSSbuf = *tempTiling;
    }
}

__aicore__ constexpr MatmulConfig GetDEFAULT_Config()
{
    auto mmCfg = CFG_NORM;
    mmCfg.iterateMode = IterateMode::ITERATE_MODE_DEFAULT;

    return mmCfg;
}

TEST_F(TestMatmulServerC310, GetOffsetSizeDefault) {
    static constexpr MatmulConfig davidConfig = GetDEFAULT_Config();
    MatmulService<A_TYPE, B_TYPE, C_TYPE_UB, BIAS_TYPE, davidConfig, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;

    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);

    KFC_Enum funID = KFC_Enum::MMFUN_ITERATE;
    uint32_t sync = 0;
    uint64_t offsetSize = 0;
    uint32_t enSequentialWrite = 0;
    bool hasSetWorkspace = true;
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    MatmulConfigParams body;
    mmServer.GetOffsetSize(&body, funID, sync, offsetSize, enSequentialWrite, hasSetWorkspace);
}

TEST_F(TestMatmulServerC310, IterateSetMessage) {
    using A_TYPE_NORMAL_SCALAR_t = MatmulType<AscendC::TPosition::GM, CubeFormat::SCALAR, half, true, LayoutMode::NORMAL>;
    static constexpr MatmulConfig IterateCFG = GetN_BATCH_Config();
    MatmulService<A_TYPE_NORMAL_SCALAR_t, B_TYPE_NORMAL_SCALAR, C_TYPE, BIAS_TYPE, IterateCFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmIteServer;
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tilingIterate;
    tilingParams.GetTiling(tilingIterate);

    KfcMsg kfcInitMsgI;
    MSG_POS TilingInfo *tilingSSbuf = reinterpret_cast<MSG_POS TilingInfo *>(GetTilingAddr(GetSubBlockIdxImpl()));
    tilingSSbuf->valid = 1;
    auto tempTilingSSbuf = reinterpret_cast<MSG_POS uint64_t*>(&(tilingSSbuf->tCubeTiling));
    auto tempTiling = reinterpret_cast<uint64_t*>(&tilingIterate);
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
        *tempTilingSSbuf = *tempTiling;
    }
    mmIteServer.Init(&kfcInitMsgI);

    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    bool isTransposeA = true;
    kfcMsg.body.isTransA = static_cast<uint32_t>(isTransposeA);
    kfcMsg.body.setTensorA = 1;
    kfcMsg.body.isFirstIter = 1;
    MatmulConfigParams bodyI = kfcMsg.body;
    bodyI.singleM = tilingParams.singleCoreM_;
    bodyI.singleN = tilingParams.singleCoreN_;
    bodyI.singleK = tilingParams.singleCoreK_;
    bodyI.setTail = 0;
    mmIteServer.IterateSetMessage(&kfcMsg, &bodyI);
}

TEST_F(TestMatmulServerC310, GetLocalTensor) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mmServer;
    uint64_t ubAddr = 0x1111;
    uint64_t ubSize = 512U;
    const auto &ubLocal = mmServer.GetLocalTensor<typename A_TYPE::T, TPosition::VECCALC>(ubAddr, ubSize);
    uint64_t tscmAddr = 0x2222;
    uint64_t tscmSize = 256U;
    const auto &tscmLocal = mmServer.GetLocalTensor<typename A_TYPE::T, TPosition::TSCM>(tscmAddr, tscmSize);

    ASSERT_TRUE((uint64_t)(ubLocal.GetPhyAddr()) == ubAddr);
    ASSERT_TRUE((uint64_t)(tscmLocal.GetPhyAddr()) ==
                (uint64_t)(GetTPipePtr()->GetBaseAddr((uint8_t)(TPosition::TSCM))) + tscmAddr);
}

TEST_F(TestMatmulServerC310, GetTensorC) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    ASSERT_TRUE(!mmServer.GetTensorC(&kfcMsg));
}

TEST_F(TestMatmulServerC310, GetMsgFromSSbuf) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    MatmulConfigParams body;
    mmServer.GetMsgFromSSbuf(&kfcMsg, body);
}

TEST_F(TestMatmulServerC310, SetQuantVector_quantmode1) {
    MatmulService<A_TYPE_NORMAL_SCALAR, B_TYPE_NORMAL_SCALAR, C_TYPE_NORMAL_SCALAR, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    kfcMsg.body.setQuant = 1;
    kfcMsg.body.quantMode = 1;
    auto body = kfcMsg.body;
    mmServer.SetQuantVector(&body);
}

TEST_F(TestMatmulServerC310, SetQuantVector_quantmode2) {
    MatmulService<A_TYPE_NORMAL_SCALAR, B_TYPE_NORMAL_SCALAR, C_TYPE_NORMAL_SCALAR,
        BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    kfcMsg.body.setQuant = 1;
    kfcMsg.body.quantMode = 2;   // 2 is quantmode xx
    auto body = kfcMsg.body;
    mmServer.SetQuantVector(&body);
}

TEST_F(TestMatmulServerC310, SetUserDefInfo) {
    MatmulService<A_TYPE_NORMAL_SCALAR, B_TYPE_NORMAL_SCALAR, C_TYPE_NORMAL_SCALAR, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    mmServer.SetUserDefInfo(&kfcMsg);
}

TEST_F(TestMatmulServerC310, SetSelfDefineData) {
    MatmulService<A_TYPE_NORMAL_SCALAR, B_TYPE_NORMAL_SCALAR, C_TYPE_NORMAL_SCALAR,
        BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.userInfoType = 1;
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    auto body = kfcMsg.body;
    mmServer.SetSelfDefineData(&kfcMsg, &body);
}

TEST_F(TestMatmulServerC310, SetSelfDefineData_failed) {
    MatmulService<A_TYPE_NORMAL_SCALAR, B_TYPE_NORMAL_SCALAR, C_TYPE_NORMAL_SCALAR,
        BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.userInfoType = 0;
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    auto body = kfcMsg.body;
    mmServer.SetSelfDefineData(&kfcMsg, &body);
}

TEST_F(TestMatmulServerC310, SkipMsg) {
    MatmulService<A_TYPE_IBSHARE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    auto funID = KFC_Enum::MMFUN_ITERATE_ALL;
    bool freeMsg = true;
    int lastMsgId = 0;
    int subBlockID = 1;
    mmServer.SkipMsg(funID, freeMsg, lastMsgId, subBlockID);
}

TEST_F(TestMatmulServerC310, SkipMsg_dIBshare) {
    MatmulService<A_TYPE_IBSHARE, B_TYPE_IBSHARE, C_TYPE, BIAS_TYPE,
        CFG_IBSHARE_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;
    auto funID = KFC_Enum::MMFUN_ITERATE_ALL;
    bool freeMsg = true;
    int lastMsgId = 0;
    int subBlockID = 1;
    mmServer.SkipMsg(funID, freeMsg, lastMsgId, subBlockID);
}

TEST_F(TestMatmulServerC310, SetTensorScaleA) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServerA;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServerA.Init(&kfcInitMsg);
    KfcMsg kfcMsgA;
    uint8_t cGM[2048] = {0};
    kfcMsgA.body.cAddr = (uint64_t)(cGM);
    kfcMsgA.body.setQuant = 1;
    kfcMsgA.body.quantMode = 1;
    kfcMsgA.body.quantAddr = NUM_FIFTYSIX;
    auto body = kfcMsgA.body;
    mmServerA.SetTensorScaleA(body);
}

TEST_F(TestMatmulServerC310, SetTensorScaleB) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;
    KfcMsg kfcInitMsg;
    setTiling();
    mmServer.Init(&kfcInitMsg);
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    kfcMsg.body.setQuant = 1;
    kfcMsg.body.quantMode = 1;
    kfcMsg.body.quantAddr = NUM_FIFTYSIX;
    auto body = kfcMsg.body;
    mmServer.SetTensorScaleB(body);
}

TEST_F(TestMatmulServerC310, ProcessEnd) {
    MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmServer;
    KfcMsg kfcMsg;
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    auto funID = KFC_Enum::MMFUN_END;
    mmServer.Process(&kfcMsg, funID);
}

TEST_F(TestMatmulServerC310, TscmInputSetAllTensor) {
    MatmulService<A_MX_TSCM_TYPE, B_MX_TSCM_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulWithScalePolicy> mmServer;
    KfcMsg kfcInitMsg;
    TilingParams tilingParams = {1, 304, 384, 128, 304, 384, 128, 128, 256, 128, 3, 2, 3, 2, 1, 1, 0, 0};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mmServer.Init(&kfcInitMsg);
    int32_t m = 304;
    int32_t n = 384;
    int32_t k = 128;
    int32_t scaleVal = 32;
    uint8_t aGm[m*k] = {0};
    uint8_t bGm[n*k] = {0};
    uint8_t scaleAGM[m*k/scaleVal] = {0};
    uint8_t scaleBGM[k*n/scaleVal] = {0};
    KfcMsg kfcMsg;
    MsgTmpPos MatmulConfigParams* body = &(kfcMsg.body);
    body->aAddr = (uint64_t)(aGm);
    body->bAddr = (uint64_t)(bGm);
    body->quantAddr = (uint64_t)(scaleAGM);
    body->quantScalar = (uint64_t)(scaleBGM);
    mmServer.SetTensorA(body);
    mmServer.SetTensorB(body);
    mmServer.SetTensorScaleA(*body);
    mmServer.SetTensorScaleB(*body);
}

TEST_F(TestMatmulServerC310, IterateNBatchBSNGD) {
    static constexpr MatmulConfig nBatchCFG = GetBmmNBatchConfig();
    MatmulService<A_TYPE_BSNGD, B_TYPE_BSNGD, C_TYPE_BSNGD, BIAS_TYPE, nBatchCFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;

    KfcMsg kfcInitMsg;
    setBatchTiling();
    mmServer.Init(&kfcInitMsg);

    KfcMsg kfcMsg;
    uint8_t cGM[131072] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    kfcMsg.body.enSequentialWrite = false;
    kfcMsg.body.sync = true;
    kfcMsg.body.batchLoop = 2;
    kfcMsg.body.batchA = 2;
    kfcMsg.body.batchB = 2;
    kfcMsg.body.matrixStrideA = 0;
    kfcMsg.body.matrixStrideB = 0;
    kfcMsg.body.matrixStrideC = 0;
    kfcMsg.body.enPartialSum = false;
    kfcMsg.body.enAtomic = (uint8_t)(0);
    kfcMsg.body.setBatch = 1;
    kfcMsg.body.waitIterateBatch = false;
    __gm__ auto *body = &(kfcMsg.body);
    uint32_t cntIterator = 0;
    mmServer.StartIterateNBatch(body, cntIterator);
    EXPECT_EQ(cntIterator, 2);
}

TEST_F(TestMatmulServerC310, IterateNBatchBNS1S2) {
    static constexpr MatmulConfig nBatchCFG = GetBmmNBatchConfig();
    MatmulService<A_TYPE_BNGS1S2, B_TYPE_BNGS1S2, C_TYPE_BNGS1S2, BIAS_TYPE, nBatchCFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, BatchCustomMatmulPolicy> mmServer;

    KfcMsg kfcInitMsg;
    setBatchTiling();
    mmServer.Init(&kfcInitMsg);

    KfcMsg kfcMsg;
    uint8_t cGM[131072] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    kfcMsg.body.enSequentialWrite = false;
    kfcMsg.body.sync = true;
    kfcMsg.body.batchLoop = 2;
    kfcMsg.body.batchA = 2;
    kfcMsg.body.batchB = 2;
    kfcMsg.body.matrixStrideA = 0;
    kfcMsg.body.matrixStrideB = 0;
    kfcMsg.body.matrixStrideC = 0;
    kfcMsg.body.enPartialSum = false;
    kfcMsg.body.enAtomic = (uint8_t)(0);
    kfcMsg.body.setBatch = 1;
    kfcMsg.body.waitIterateBatch = false;
    __gm__ auto *body = &(kfcMsg.body);
    uint32_t cntIterator = 0;
    mmServer.StartIterateNBatch(body, cntIterator);
    EXPECT_EQ(cntIterator, 2);
}
} // namespace AscendC
