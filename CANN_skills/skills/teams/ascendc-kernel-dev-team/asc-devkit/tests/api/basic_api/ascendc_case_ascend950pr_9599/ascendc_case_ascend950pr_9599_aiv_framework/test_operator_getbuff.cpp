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
#include "kernel_utils.h"

namespace AscendC {
class KernelAdd {
public:
    __aicore__ inline KernelAdd()
    {}
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z)
    {
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ float *)x, 1024);
        yGm.SetGlobalBuffer((__gm__ float *)y, 1024);
        zGm.SetGlobalBuffer((__gm__ float *)z, 1024);
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(tmpBufInX[0], 1024 * sizeof(float));
        pipe.InitBuffer(tmpBufInY[0], 1024 * sizeof(float));
        pipe.InitBuffer(tmpBufOutZ[0], 1024 * sizeof(float));

        pipe.InitBuffer(tmpBufInX[1], 1024 * sizeof(float));
        pipe.InitBuffer(tmpBufInY[1], 1024 * sizeof(float));
        pipe.InitBuffer(tmpBufOutZ[1], 1024 * sizeof(float));
    }
    // 极简模式最少使用同步指令 & 使能double buffer
    __aicore__ inline void ProcessSimple()
    {
        constexpr int32_t loopCount = 8;
        // tiling strategy, pipeline parallel
        uint8_t mutexId = 0;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i % 2 == 0) {
                mutexId = 0;
            } else {
                mutexId = 1;
            }
            CopyInSimple(i, mutexId);
            ComputeSimple(i, mutexId);
            CopyOutSimple(i, mutexId);
        }
    }

    // 模拟Enque & Deque & Alloc & Free & 使能double buffer
    __aicore__ inline void ProcessFull()
    {
        constexpr int32_t loopCount = 8;
        // tiling strategy, pipeline parallel
        MutexID mutexId;
        MutexID mutexId1 = GetTPipePtr()->AllocMutexID();
        MutexID mutexId2 = AllocMutexID();
        EXPECT_EQ(mutexId1, 0);
        EXPECT_EQ(mutexId2, 1);
        for (int32_t i = 0; i < loopCount; i++) {
            if (i % 2 == 0) {
                mutexId = mutexId1;
            } else {
                mutexId = mutexId2;
            }
            CopyInSimple(i, mutexId);
            ComputeSimple(i, mutexId);
            CopyOutSimple(i, mutexId);
        }
        GetTPipePtr()->ReleaseMutexID(mutexId1);
        ReleaseMutexID(mutexId2);
        EXPECT_EQ(AllocMutexID(), 0);
    }

private:
    __aicore__ inline void CopyInSimple(int32_t progress, uint8_t mutexId)
    {
        LocalTensor<float> xLocal = tmpBufInX[mutexId].Get<float>();
        LocalTensor<float> yLocal = tmpBufInY[mutexId].Get<float>();
        // copy progress_th tile from global tensor to local tensor
        Mutex::Lock<PIPE_MTE2>(mutexId);
        DataCopy(xLocal, xGm[progress * 1024], 1024);
        DataCopy(yLocal, yGm[progress * 1024], 1024);
        Mutex::Unlock<PIPE_MTE2>(mutexId);
    }
    __aicore__ inline void ComputeSimple(int32_t progress, uint8_t mutexId)
    {
        LocalTensor<float> xLocal = tmpBufInX[mutexId].Get<float>();
        LocalTensor<float> yLocal = tmpBufInY[mutexId].Get<float>();
        LocalTensor<float> zLocal = tmpBufOutZ[mutexId].Get<float>();

        Mutex::Lock<PIPE_V>(mutexId);
        Add(zLocal, yLocal, xLocal, 1024);
        Mutex::Unlock<PIPE_V>(mutexId);
    }
    __aicore__ inline void CopyOutSimple(int32_t progress, uint8_t mutexId)
    {
        LocalTensor<float> zLocal = tmpBufOutZ[mutexId].Get<float>();
        Mutex::Lock<PIPE_MTE3>(mutexId);
        DataCopy(zGm[progress * 1024], zLocal, 1024);
        Mutex::Unlock<PIPE_MTE3>(mutexId);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TBuf<> tmpBufInX[2];
    TBuf<> tmpBufInY[2];
    TBuf<> tmpBufOutZ[2];
    // create queue for output, in this case depth is equal to buffer num
    GlobalTensor<float> xGm, yGm, zGm;
};
}

__global__ __aicore__ void testGetBuff(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    AscendC::KernelAdd op;
    op.Init(x, y, z);
    op.ProcessFull();
}

struct GetBuffParams {
    int32_t dataSize;
    int32_t sizeOfT;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *);
};

class GetBuffTestsuite : public testing::Test, public testing::WithParamInterface<GetBuffParams> {
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

INSTANTIATE_TEST_CASE_P(
    TEST_OPEARATION_GETBUFF, GetBuffTestsuite, ::testing::Values(GetBuffParams{16384, 4, testGetBuff}));

TEST_P(GetBuffTestsuite, testGetBuff)
{
    auto param = GetParam();
    uint8_t xGm[param.dataSize * param.sizeOfT] = {0x00};
    uint8_t yGm[param.dataSize * param.sizeOfT] = {0x00};
    uint8_t zGm[param.dataSize * param.sizeOfT] = {0x00};
    param.cal_func(zGm, xGm, yGm);

    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(zGm[i], 0x00);
    }
}