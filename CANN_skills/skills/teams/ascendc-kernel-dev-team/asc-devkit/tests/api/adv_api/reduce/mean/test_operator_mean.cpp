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

namespace AscendC {

template <typename T, typename accType> class KernelMean {
public:
    __aicore__ inline KernelMean() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, uint32_t outter, uint32_t inner, uint32_t n)
    {
        elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);         // half=16 float=8
        elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
        meanParams.outter = outter;
        meanParams.inner = inner;
        meanParams.n = n;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
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
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, meanParams.outter * meanParams.inner);
        inQueueX.EnQue(srcLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueY.AllocTensor<T>();
        T scalar(0);
        Duplicate<T>(dstLocal, scalar, (meanParams.outter * sizeof(T) + ONE_BLK_SIZE -1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T));
        Mean<T, accType>(dstLocal, srcLocal, meanParams);
        outQueueY.EnQue<T>(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueY.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, (meanParams.outter * sizeof(T) + ONE_BLK_SIZE -1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T));
        outQueueY.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> tmplocalBuf;

    uint32_t elementNumPerBlk = 0;
    uint32_t elementNumPerRep = 0;
    MeanParams meanParams;
};

}  // namespace AscendC


template <typename T, typename accType>
__global__ __aicore__ void main_Mean_test(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t outter, uint32_t inner, uint32_t n)
{
    AscendC::KernelMean<T, accType> op;
    op.Init(srcGm, dstGm, outter, inner, n);
    op.Process();
}

struct MeanTestParams {
    MeanParams meanParams;
    uint32_t dataTypeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
};

class MeanTestSuite : public testing::Test, public testing::WithParamInterface<MeanTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MeanTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MeanTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp()
    {}
    virtual void TearDown()
    {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SUM, MeanTestSuite,
    ::testing::Values(
        MeanTestParams { {1, 32, 2}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 64, 64}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 128, 128}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 256, 256}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 2048, 2048}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 8192, 8192}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {1, 31072, 31072}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {2, 32, 32}, sizeof(half), main_Mean_test<half, float>},
        MeanTestParams { {4, 1024, 1022}, sizeof(half), main_Mean_test<half, float>},

        MeanTestParams { {1, 32, 2}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 64, 64}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 128, 128}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 256, 256}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 2048, 2048}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 8192, 8192}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {1, 93456, 93456}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {2, 32, 32}, sizeof(half), main_Mean_test<half, half>},
        MeanTestParams { {4, 1024, 1022}, sizeof(half), main_Mean_test<half, half>},

        MeanTestParams { {1, 32, 2}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 64, 64}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 128, 128}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 256, 256}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 2048, 2048}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 8192, 8192}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {1, 46368, 46368}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {2, 32, 32}, sizeof(float), main_Mean_test<float, float>},
        MeanTestParams { {4, 1024, 1022}, sizeof(float), main_Mean_test<float, float>}
        ));

TEST_P(MeanTestSuite, MeanTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.meanParams.outter * param.meanParams.inner * param.dataTypeSize] {0x00};
    uint8_t dstGm[(param.meanParams.outter * param.dataTypeSize + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE] {0x00};
    param.cal_func(srcGm, dstGm, param.meanParams.outter, param.meanParams.inner, param.meanParams.n);
    for (int32_t i = 0; i < param.meanParams.outter; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
