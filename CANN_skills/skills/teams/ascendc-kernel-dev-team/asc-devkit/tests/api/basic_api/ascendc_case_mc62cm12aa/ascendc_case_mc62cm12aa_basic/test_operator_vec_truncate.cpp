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
#include <type_traits>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
template <typename T, RoundMode roundMode>
class KernelVecTruncate {
public:
    __aicore__ inline KernelVecTruncate()
    {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t dataSizeIn, uint32_t calCountIn)
    {
        calCount = calCountIn;
        const int alignSize = 32 / sizeof(T);
        dataSize = CeilDivision(dataSizeIn, alignSize) * alignSize;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);

        pipe.InitBuffer(inQueue, 1, dataSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, dataSize);
        inQueue.EnQue<T>(srcLocal);
    }

    __aicore__ inline void Compute()
    {
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint32_t repeatTimes = dataSize / repeatElm;
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        const T Zero = 0;
        Duplicate(dstLocal, Zero, dataSize);
        Duplicate(dstLocal, Zero, dataSize);
     
        Truncate<T, roundMode>(dstLocal, srcLocal, calCount);
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t calCount;
    uint32_t dataSize;
};

struct VecTruncateParams {
    void (*CallFunc)();
};
 
template<typename T, RoundMode roundMode>
void RunCase() {
    int byte_size = sizeof(T);
    int data_size = 256;
    int mask = 256;
    uint8_t dstGm[data_size * byte_size] = {0};
    uint8_t srcGm[data_size * byte_size] = {0};
 
    KernelVecTruncate<T, roundMode> op;
    op.Init(dstGm, srcGm, data_size, mask);
    op.Process();
}
 
class VecTruncateTestSuite : public testing::Test, public testing::WithParamInterface<VecTruncateParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(VecTruncateTestCases, VecTruncateTestSuite,
    ::testing::Values(VecTruncateParams { RunCase<float, RoundMode::CAST_RINT>},
                      VecTruncateParams { RunCase<float, RoundMode::CAST_ROUND>},
                      VecTruncateParams { RunCase<float, RoundMode::CAST_FLOOR>},
                      VecTruncateParams { RunCase<float, RoundMode::CAST_CEIL>},
                      VecTruncateParams { RunCase<float, RoundMode::CAST_TRUNC>},

                      VecTruncateParams { RunCase<half, RoundMode::CAST_RINT>},
                      VecTruncateParams { RunCase<half, RoundMode::CAST_ROUND>},
                      VecTruncateParams { RunCase<half, RoundMode::CAST_FLOOR>},
                      VecTruncateParams { RunCase<half, RoundMode::CAST_CEIL>},
                      VecTruncateParams { RunCase<half, RoundMode::CAST_TRUNC>},

                      VecTruncateParams { RunCase<bfloat16_t, RoundMode::CAST_RINT>},
                      VecTruncateParams { RunCase<bfloat16_t, RoundMode::CAST_ROUND>},
                      VecTruncateParams { RunCase<bfloat16_t, RoundMode::CAST_FLOOR>},
                      VecTruncateParams { RunCase<bfloat16_t, RoundMode::CAST_CEIL>},
                      VecTruncateParams { RunCase<bfloat16_t, RoundMode::CAST_TRUNC>}
                      ));
 
TEST_P(VecTruncateTestSuite, VecTruncateTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}