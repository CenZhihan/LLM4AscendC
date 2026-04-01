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


using namespace std;
using namespace AscendC;

namespace AscendC {

template <typename dataType, bool isReuseSrc = false, uint8_t expandLevel = 10>
class KernelExpHighPrecision {
public:
    __aicore__ inline KernelExpHighPrecision() {}

    __aicore__ inline void Init(GM_ADDR inputXGm, GM_ADDR outputGm, uint32_t totalLength, uint32_t calCount,
        uint32_t mode)
    {
        uint32_t oneBlockNum = 32 / sizeof(dataType);
        totalLength = (totalLength + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
        this->totalLength = totalLength;
        this->calCount = calCount;
        this->mode = mode;
        this->dataFormat = DataFormat::ND;

        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputXGm), totalLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputGm), totalLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * totalLength);
        pipe.InitBuffer(outQueue, 1, sizeof(dataType) * totalLength);
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
        LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();

        DataCopy(inputXLocal, inputXGlobal, totalLength);

        inQueueX.EnQue(inputXLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();

        LocalTensor<uint8_t> stackBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();

        // mode 0: no input tmpBuffer, mode 1: tmpBuffer
        if (mode == 0) {
            Exp<dataType, expandLevel, isReuseSrc>(outputLocal, inputXLocal, calCount);
        } else {
            LocalTensor<uint8_t> stackBuffer;
            bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
            stackBuffer.SetSize(stackBufferSize);
            Exp<dataType, expandLevel, isReuseSrc>(outputLocal, inputXLocal, stackBuffer, calCount);
        }

        outQueue.EnQue<dataType>(outputLocal);

        inQueueX.FreeTensor(inputXLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();

        DataCopy(outputGlobal, outputLocal, totalLength);

        outQueue.FreeTensor(outputLocal);
    }

private:
    GlobalTensor<dataType> inputXGlobal;
    GlobalTensor<dataType> outputGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint32_t totalLength;
    uint32_t calCount;
    uint32_t mode;
    uint32_t stackBufferSize = 0;
    DataFormat dataFormat;
};
} // namespace AscendC

template <typename dataType, bool isReuseSrc = false, uint8_t expandLevel = 10>
__global__ __aicore__ void kernel_exphighprecision_operator(GM_ADDR inputXGm, GM_ADDR outputGm, uint32_t totalLength,
    uint32_t calCount, uint32_t mode)
{
    AscendC::KernelExpHighPrecision<dataType, isReuseSrc, expandLevel> op;
    op.Init(inputXGm, outputGm, totalLength, calCount, mode);
    op.Process();
}

struct ExpHighPrecisionTestParams {
    uint32_t totalLength;
    uint32_t calCount;
    uint32_t mode;
    uint32_t typeSize;
    void (*calFunc) (uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
};

class ExpHighPrecisionTestsuite : public testing::Test, public testing::WithParamInterface<ExpHighPrecisionTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_EXPHIGHPRECISION, ExpHighPrecisionTestsuite,
    ::testing::Values(
    ExpHighPrecisionTestParams {1024, 1024, 0, sizeof(half),  kernel_exphighprecision_operator<half,  true, 10> },
    ExpHighPrecisionTestParams {1024, 1000, 0, sizeof(half),  kernel_exphighprecision_operator<half,  false, 10>},
    ExpHighPrecisionTestParams {1024, 1024, 0, sizeof(float), kernel_exphighprecision_operator<float, true, 10> },
    ExpHighPrecisionTestParams {1024, 1000, 0, sizeof(float), kernel_exphighprecision_operator<float, false, 10>},
    ExpHighPrecisionTestParams {1024, 1024, 1, sizeof(half),  kernel_exphighprecision_operator<half,  true, 10> }
));

TEST_P(ExpHighPrecisionTestsuite, ExpHighPrecisionOpTestCase)
{
    auto param = GetParam();

    uint32_t totalLength = param.totalLength;
    uint32_t calCount = param.calCount;
    uint32_t mode = param.mode;
    uint32_t typeSize = param.typeSize;

    uint32_t oneBlockNum = 32 / typeSize;
    totalLength = (totalLength + oneBlockNum - 1) / oneBlockNum * oneBlockNum;

    uint8_t inputXGm[totalLength * typeSize] { 0x00 };
    uint8_t outputGm[totalLength * typeSize] { 0x00 };

    param.calFunc(inputXGm, outputGm, totalLength, calCount, mode);
}