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

using namespace std;
using namespace AscendC;

class TEST_VPADDING : public testing::Test {
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

template <typename T> class KernelVpadding {
public:
    __aicore__ inline KernelVpadding() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, int64_t mask[2],
        uint32_t repeatTimes, int64_t repeatParams[4])
    {
        this->totalLength = totalLength;
        this->mask[0] = mask[0];
        this->mask[1] = mask[1];
        this->repeatTimes = repeatTimes;
        this->repeatParams.dstBlkStride = repeatParams[0];
        this->repeatParams.srcBlkStride = repeatParams[1];
        this->repeatParams.dstRepStride = repeatParams[2];
        this->repeatParams.srcRepStride = repeatParams[3];

        xGm.SetGlobalBuffer((__gm__ T *)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, this->totalLength);
        pipe.InitBuffer(inQueueX, 1, this->totalLength * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, this->totalLength * sizeof(T));
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
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm, this->totalLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if (mask[0] == 0 && mask[1] == 0) {
            VectorPadding(yLocal, xLocal, 0, false, totalLength);
        } else if (mask[0] == 0 && mask[1] != 0) {
            VectorPadding(yLocal, xLocal, 1, true, mask[1], repeatTimes, repeatParams);
        } else {
            VectorPadding(yLocal, xLocal, 2, false, mask, repeatTimes, repeatParams);
        }
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm, yLocal, this->totalLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueueY;
    GlobalTensor<T> xGm, yGm;
    uint32_t totalLength;
    uint64_t mask[2];
    uint8_t repeatTimes;
    UnaryRepeatParams repeatParams;
};

#define VEC_VPADDING_TESTCASE(mask1, mask2, type)                                   \
    TEST_F(TEST_VPADDING, VPADDING##_##mask1##_##mask2##_##type##_##Case)           \
    {                                                                               \
        KernelVpadding<type> op;                                                    \
        int64_t dataSize = 128;                                                     \
        int64_t tmpMask1 = mask1;                                                  \
        int64_t tmpMask2 = mask2;                                                  \
        int64_t mask[] = {tmpMask1, tmpMask2};                                    \
        int64_t repeatParams[] = {1, 1, 8, 8};                                      \
        uint8_t inputGm[dataSize * sizeof(type)];                                   \
        uint8_t outputGm[dataSize * sizeof(type)];                                  \
        op.Init(outputGm, inputGm, dataSize, mask, 1, repeatParams);                \
        op.Process();                                                               \
    }
VEC_VPADDING_TESTCASE(0, 0, uint16_t);
VEC_VPADDING_TESTCASE(0, 128, uint16_t);
VEC_VPADDING_TESTCASE(65535, 65535, uint16_t);

VEC_VPADDING_TESTCASE(0, 0, uint32_t);
VEC_VPADDING_TESTCASE(0, 64, uint32_t);
VEC_VPADDING_TESTCASE(65535, 0, uint32_t);