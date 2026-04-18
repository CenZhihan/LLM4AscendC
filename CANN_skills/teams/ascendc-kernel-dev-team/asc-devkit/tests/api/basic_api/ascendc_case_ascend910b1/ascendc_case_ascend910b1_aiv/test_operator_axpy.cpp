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
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

template <typename T, typename U>
void AxpyKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    TQue<TPosition::VECIN, 1> vecInQue;
    TQue<TPosition::VECIN, 1> vecOutQue;
    TQue<TPosition::VECIN, 1> vecTmpQue;
    GlobalTensor<U> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
    tpipe.InitBuffer(vecInQue, 1, dataSize * sizeof(U));
    tpipe.InitBuffer(vecOutQue, 1, dataSize * sizeof(T));
    if (sizeof(U) == sizeof(float)) { 
        tpipe.InitBuffer(vecTmpQue, 1, dataSize * sizeof(float));
    } else {
        tpipe.InitBuffer(vecTmpQue, 1, dataSize * 4 * sizeof(half));
    }
    LocalTensor<U> inputLocal = vecInQue.AllocTensor<U>();
    LocalTensor<T> outputLocal = vecOutQue.AllocTensor<T>();
    LocalTensor<uint8_t> tmpLocal = vecTmpQue.AllocTensor<uint8_t>();

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    AscendCUtils::SetMask<uint8_t>(128);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    U scalar = 4;

    Axpy<T, U, false>(outputLocal, inputLocal, scalar, tmpLocal, dataSize);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    PipeBarrier<PIPE_ALL>();
    vecInQue.FreeTensor(inputLocal);
    vecOutQue.FreeTensor(outputLocal);
    vecTmpQue.FreeTensor(tmpLocal);
}

struct AxpyTestParams {
    int32_t dataSize;
    int32_t dataBitSize;
    void (*calFunc)(uint8_t*, uint8_t*, int32_t);
};

class AxpyTestsuite : public testing::Test, public testing::WithParamInterface<AxpyTestParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_AXPY, AxpyTestsuite,
    ::testing::Values(AxpyTestParams { 256, 2, AxpyKernel<half, half> },
    AxpyTestParams { 256, 4, AxpyKernel<float, half> },
    AxpyTestParams { 256, 4, AxpyKernel<float, float> }));

TEST_P(AxpyTestsuite, AxpyTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.dataBitSize] = {0};
    uint8_t dstGm[param.dataSize * param.dataBitSize] = {0};

    param.calFunc(srcGm, dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
