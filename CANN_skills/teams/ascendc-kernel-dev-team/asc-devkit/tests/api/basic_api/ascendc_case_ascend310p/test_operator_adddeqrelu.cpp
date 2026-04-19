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

class TEST_ADDDEQRELU : public testing::Test {
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

class KernelAddDeqRelu {
public:
    __aicore__ inline KernelAddDeqRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, int64_t mask[2],
        uint32_t repeatTimes, int64_t repeatParams[6])
    {
        this->totalLength = totalLength;
        this->mask[0] = mask[0];
        this->mask[1] = mask[1];
        this->repeatTimes = repeatTimes;
        this->repeatParams.dstBlkStride = repeatParams[0];
        this->repeatParams.src0BlkStride = repeatParams[1];
        this->repeatParams.src1BlkStride = repeatParams[2];
        this->repeatParams.dstRepStride = repeatParams[3];
        this->repeatParams.src0RepStride = repeatParams[4];
        this->repeatParams.src1RepStride = repeatParams[5];

        xGm.SetGlobalBuffer((__gm__ int32_t *)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ int32_t *)y, this->totalLength);
        zGm.SetGlobalBuffer((__gm__ half *)z, this->totalLength);
        pipe.InitBuffer(inQueueX, 1, this->totalLength * sizeof(int32_t));
        pipe.InitBuffer(inQueueY, 1, this->totalLength * sizeof(int32_t));
        pipe.InitBuffer(outQueueZ, 1, this->totalLength * sizeof(half));
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
        LocalTensor<int32_t> xLocal = inQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> yLocal = inQueueY.AllocTensor<int32_t>();
        DataCopy(xLocal, xGm, this->totalLength);
        DataCopy(yLocal, yGm, this->totalLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> xLocal = inQueueX.DeQue<int32_t>();
        LocalTensor<int32_t> yLocal = inQueueY.DeQue<int32_t>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        half scale = 0.1;
        SetDeqScale(scale);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (mask[0] == 0 && mask[1] == 0) {
            AddDeqRelu(zLocal, xLocal, yLocal, totalLength);
        } else if (mask[0] == 0 && mask[1] != 0) {
            AddDeqRelu(zLocal, xLocal, yLocal, mask[1], repeatTimes, repeatParams);
        } else {
            uint64_t tmp0 = 1;
            for (int i = 0; i < (mask[0] - 1); i++) {
                tmp0 *= 2;
            }
            mask[0] = tmp0 == 0 ? tmp0 : tmp0 * 2 - 1;
            tmp0 = 1;
            for (int i = 0; i < (mask[1] - 1); i++) {
                tmp0 *= 2;
            }
            mask[1] = tmp0 == 0 ? tmp0 : tmp0 * 2 - 1;
            AddDeqRelu(zLocal, xLocal, yLocal, mask, repeatTimes, repeatParams);
        }
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm, zLocal, this->totalLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, 1> outQueueZ;
    GlobalTensor<int32_t> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<half> zGm;
    uint32_t totalLength;
    uint64_t mask[2];
    uint8_t repeatTimes;
    BinaryRepeatParams repeatParams;
};

#define VEC_ADDDEQRELU_TESTCASE(mask1, mask2)                                   \
    TEST_F(TEST_ADDDEQRELU, ADDDEQRELU##_##mask1##_##mask2##_##Case)                 \
    {                                                                           \
        KernelAddDeqRelu op;                                                    \
        int64_t dataSize = 64;                                                 \
        int64_t tmpMask1 = mask1;                                              \
        int64_t tmpMask2 = mask2;                                              \
        int64_t mask[] = {tmpMask1, tmpMask2};                              \
        int64_t repeatParams[] = {1, 1, 1, 4, 8, 8};                          \
        uint8_t input0Gm[dataSize * sizeof(int32_t)];                           \
        uint8_t input1Gm[dataSize * sizeof(int32_t)];                           \
        uint8_t outputGm[dataSize * sizeof(half)];                              \
        op.Init(outputGm, input0Gm, input1Gm, dataSize, mask, 1, repeatParams); \
        op.Process();                                                           \
    }
VEC_ADDDEQRELU_TESTCASE(0, 0);
VEC_ADDDEQRELU_TESTCASE(0, 64);