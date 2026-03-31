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
#define private public
#define protected public
#include "kernel_operator.h"
#include <iostream>
using namespace std;
using namespace AscendC;

constexpr uint32_t WEL_UP_BLOCK_SIZE = 32;
constexpr WelfordUpdateConfig WELFORD_UPDATE_ENABLE_INPLACE_CFG = {true};
constexpr WelfordUpdateConfig WELFORD_UPDATE_UNENABLE_INPLACE_CFG = {false};
namespace TEST_CASE {
constexpr uint32_t WEL_UP_REP_SIZE = 256;
constexpr uint32_t WEL_UP_FLOAT_SIZE = 256 / sizeof(float);

bool GetWelfordUpdateMaxMinTmpSize(const int32_t rnLength, const int32_t abLength, const uint32_t typeSizeT,
    const uint32_t typeSizeU, const bool isReuseSource, const bool isInplace, uint32_t& maxValue, uint32_t& minValue)
{
    if (typeSizeT == sizeof(uint16_t)) {
        minValue = 0x3 * WEL_UP_REP_SIZE;
    } else if (isReuseSource) {
        minValue = 1 * WEL_UP_REP_SIZE;
    } else {
        minValue = 0x2 * WEL_UP_REP_SIZE;
    }
    maxValue = (rnLength * abLength + WEL_UP_FLOAT_SIZE - 1) / WEL_UP_FLOAT_SIZE * minValue;
    return true;
}

}  // namespace TEST_CASE

template <typename T, typename U, bool isReuseSource = false, bool isInplace = false, bool tmpLocal = false>
class KernelWelfordUpdate {
public:
    __aicore__ inline KernelWelfordUpdate()
    {}
    __aicore__ inline void Init(GM_ADDR inputX_gm, GM_ADDR inputMean_gm, GM_ADDR inputVar_gm, GM_ADDR outputMean_gm,
        GM_ADDR outputVar_gm, int32_t rnLength, int32_t abLength, int32_t abComputeLength, float nRec)
    {
        m_rnLength = rnLength;
        m_abLength = abLength;
        m_abComputeLength = abComputeLength;
        m_nRec = nRec;
        bshLength = rnLength * abLength;
        inplace = isInplace;

        inputX_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputX_gm), bshLength);
        inputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(inputMean_gm), bshLength);
        inputVar_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(inputVar_gm), bshLength);
        outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(outputMean_gm), bshLength);
        outputVar_global.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(outputVar_gm), bshLength);

        pipe.InitBuffer(inQueueX, 1, bshLength * sizeof(T));
        pipe.InitBuffer(inQueueMean, 1, bshLength * sizeof(U));
        pipe.InitBuffer(inQueueVar, 1, bshLength * sizeof(U));
        pipe.InitBuffer(outQueueMean, 1, bshLength * sizeof(U));
        pipe.InitBuffer(outQueueVar, 1, bshLength * sizeof(U));

    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> inputXLocal = inQueueX.AllocTensor<T>();
        LocalTensor<U> inputMeanLocal = inQueueMean.AllocTensor<U>();
        LocalTensor<U> inputVarLocal = inQueueVar.AllocTensor<U>();

        DataCopy(inputXLocal, inputX_global, bshLength);
        DataCopy(inputMeanLocal, inputMean_global, bshLength);
        DataCopy(inputVarLocal, inputVar_global, bshLength);

        inQueueX.EnQue(inputXLocal);
        inQueueMean.EnQue(inputMeanLocal);
        inQueueVar.EnQue(inputVarLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> inputXLocal = inQueueX.DeQue<T>();
        LocalTensor<U> inputMeanLocal = inQueueMean.DeQue<U>();
        LocalTensor<U> inputVarLocal = inQueueVar.DeQue<U>();

        LocalTensor<U> outMeanLocal = outQueueMean.AllocTensor<U>();
        LocalTensor<U> outVarLocal = outQueueVar.AllocTensor<U>();

        AscendC::Duplicate<U>(outMeanLocal, (U)(0.0), bshLength);
        AscendC::Duplicate<U>(outVarLocal, (U)(0.0), bshLength);

        struct WelfordUpdateParam para = {m_rnLength, m_abLength, m_abComputeLength, m_nRec};
        if (tmpLocal) {
            TEST_CASE::GetWelfordUpdateMaxMinTmpSize(m_rnLength, m_abLength, sizeof(T), sizeof(U), isReuseSource,
                isInplace, tmpMaxBytes, tmpMinBytes);
            if (tmpMinBytes % WEL_UP_BLOCK_SIZE != 0) {
                tmpMinBytes = (tmpMinBytes + WEL_UP_BLOCK_SIZE - 1) / WEL_UP_BLOCK_SIZE * WEL_UP_BLOCK_SIZE;
            }
            pipe.InitBuffer(tmpLocalBuf, tmpMinBytes);
            LocalTensor<uint8_t> tmpLocalTensor = tmpLocalBuf.Get<uint8_t>();
            if (inplace) {
                WelfordUpdate<T, U, isReuseSource, WELFORD_UPDATE_ENABLE_INPLACE_CFG>(outMeanLocal, outVarLocal,
                    inputMeanLocal, inputVarLocal, inputXLocal, tmpLocalTensor, para);
            } else {
                WelfordUpdate<T, U, isReuseSource, WELFORD_UPDATE_UNENABLE_INPLACE_CFG>(outMeanLocal, outVarLocal,
                    inputMeanLocal, inputVarLocal, inputXLocal, tmpLocalTensor, para);
            }
        } else {
            if (inplace) {
                WelfordUpdate<T, U, isReuseSource, WELFORD_UPDATE_ENABLE_INPLACE_CFG>(outMeanLocal, outVarLocal,
                    inputMeanLocal, inputVarLocal, inputXLocal, para);
            } else {
                WelfordUpdate<T, U, isReuseSource, WELFORD_UPDATE_UNENABLE_INPLACE_CFG>(outMeanLocal, outVarLocal,
                    inputMeanLocal, inputVarLocal, inputXLocal, para);
            }
        }
        
        outQueueMean.EnQue<U>(outMeanLocal);
        outQueueVar.EnQue<U>(outVarLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueMean.FreeTensor(inputMeanLocal);
        inQueueVar.FreeTensor(inputVarLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<U> outMeanLocal = outQueueMean.DeQue<U>();
        LocalTensor<U> outVarLocal = outQueueVar.DeQue<U>();

        DataCopy(outputMean_global, outMeanLocal, bshLength);
        DataCopy(outputVar_global, outVarLocal, bshLength);

        outQueueMean.FreeTensor(outMeanLocal);
        outQueueVar.FreeTensor(outVarLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueMean;
    TQue<TPosition::VECIN, 1> inQueueVar;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueVar;

    GlobalTensor<T> inputX_global;
    GlobalTensor<U> inputMean_global;
    GlobalTensor<U> inputVar_global;
    GlobalTensor<U> outputMean_global;
    GlobalTensor<U> outputVar_global;
    TBuf<TPosition::VECCALC> tmpLocalBuf;

    uint32_t m_rnLength;
    uint32_t m_abLength;
    uint32_t m_abComputeLength;
    float m_nRec;
    uint32_t bshLength;
    bool inplace;
    uint32_t tmpMinBytes = 0;
    uint32_t tmpMaxBytes = 0;
};

template <typename T, typename U, bool isReuseSource = false, bool isFullwelfordUpdate = false, bool tmpLocal = false>
__aicore__ void main_WelfordUpdate_test(GM_ADDR srcGm, GM_ADDR inMeanGm, GM_ADDR inVarGm, GM_ADDR outMeanGm,
    GM_ADDR outVarGm, int32_t rnLength, int32_t abLength, int32_t abComputeLength, float nRec)
{
    KernelWelfordUpdate<T, U, isReuseSource, isFullwelfordUpdate, tmpLocal> op;
    op.Init(srcGm, inMeanGm, inVarGm, outMeanGm, outVarGm, rnLength, abLength, abComputeLength, nRec);
    op.Process();
}

struct WelfordUpdateTestParams {
    int64_t rnLength;
    int64_t abLength;
    int64_t abComputeLength;
    float nRec;
    uint32_t TypeSizeT;
    uint32_t TypeSizeU;
    void (*calFunc)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, int32_t, int32_t, int32_t, float);
};

class WelfordUpdateTestSuite : public testing::Test, public testing::WithParamInterface<WelfordUpdateTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "WelfordUpdateTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "WelfordUpdateTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp()
    {}
    virtual void TearDown()
    {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_WelfordUpdate, WelfordUpdateTestSuite,
    ::testing::Values(
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, true, true, true> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, true, true, false> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, true, false, true> },
    WelfordUpdateTestParams { 1, 16, 16, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, false, true, true> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, true, false, false> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, false, true, false> },
    WelfordUpdateTestParams { 1, 16, 16, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, false, false, true> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(half), sizeof(float), main_WelfordUpdate_test<half, float, false, false, false> },

    WelfordUpdateTestParams { 1, 8, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, true, true, true> },
    WelfordUpdateTestParams { 1, 16, 13, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, true, true, false> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, true, false, true> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, false, true, true> },
    WelfordUpdateTestParams { 1, 16, 16, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, true, false, false> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, false, true, false> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, false, false, true> },
    WelfordUpdateTestParams { 1, 16, 3, 0.8, sizeof(float), sizeof(float), main_WelfordUpdate_test<float, float, false, false, false> }
));

TEST_P(WelfordUpdateTestSuite, WelfordUpdateTestCase)
{
    auto param = GetParam();
    uint32_t srcSize = param.rnLength * param.abLength;
    uint8_t srcGm[srcSize * param.TypeSizeT]{0x00};
    uint8_t inMeanGm[srcSize * param.TypeSizeU]{0x00};
    uint8_t inVarGm[srcSize * param.TypeSizeU]{0x00};
    uint8_t outMeanGm[srcSize * param.TypeSizeU]{0x00};
    uint8_t outVarGm[srcSize * param.TypeSizeU]{0x00};
    param.calFunc(srcGm, inMeanGm, inVarGm, outMeanGm, outVarGm, param.rnLength, param.abLength, param.abComputeLength,
        param.nRec);
    for (int32_t i = 0; i < srcSize; i++) {
        EXPECT_EQ(outMeanGm[i], 0x00);
        EXPECT_EQ(outVarGm[i], 0x00);
    }
}

