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
#include "kernel_utils.h"
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

template <typename T>
__global__ __aicore__ void MainSyncAllTest(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ uint8_t* __restrict__ workGm, __gm__ uint8_t* __restrict__ syncGm)
{
    AscendC::TPipe tpipe;
    block_num = 1; // system numBlocks
    int32_t blockNum = block_num;
    uint32_t srcDataSize = 256;
    uint32_t perBlockDataSize = srcDataSize;

    int32_t blockIdx = 0;
    GlobalTensor<float> srcGlobal;
    GlobalTensor<float> dstGlobal;
    GlobalTensor<float> workGlobal;
    GlobalTensor<int32_t> syncGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockDataSize * sizeof(float)),
        perBlockDataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockDataSize * sizeof(float)),
        perBlockDataSize);
    workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workGm), srcDataSize);
    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncGm), 8);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, perBlockDataSize * sizeof(float));
    LocalTensor<float> inputLocal = tbuf1.Get<float>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, perBlockDataSize * sizeof(float));
    LocalTensor<float> outputLocal = tbuf2.Get<float>();

    TBuf<TPosition::VECCALC> tbuf3;
    tpipe.InitBuffer(tbuf3, FLOAT_REPEAT_SIZE);
    LocalTensor<int32_t> workLocal = tbuf3.Get<int32_t>();

    float zero(0);
    Duplicate(outputLocal, zero, perBlockDataSize);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    DataCopy(inputLocal, srcGlobal, perBlockDataSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    float scalarValue(2);
    Muls(inputLocal, inputLocal, scalarValue, perBlockDataSize);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(workGlobal[blockIdx * perBlockDataSize], inputLocal, perBlockDataSize);

    SyncAll(syncGlobal, workLocal);

    for (int i = 0; i < blockNum; i++) {
        if (i != blockIdx) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            DataCopy(outputLocal, workGlobal[i * perBlockDataSize], perBlockDataSize);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Add(inputLocal, inputLocal, outputLocal, perBlockDataSize);
        }
    }

    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(dstGlobal, inputLocal, perBlockDataSize);

    pipe_barrier(PIPE_ALL);
}

struct SyncAllTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*);
};

class SyncAllTestsuite : public testing::Test, public testing::WithParamInterface<SyncAllTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SYNCALL, SyncAllTestsuite,
    ::testing::Values(SyncAllTestParams { 4, MainSyncAllTest<float> }));

TEST_P(SyncAllTestsuite, SyncAllOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[256 * param.typeSize];
    uint8_t dstGm[256 * param.typeSize];
    uint8_t workGm[256 * param.typeSize];
    uint8_t syncGm[8 * 4];

    param.cal_func(dstGm, srcGm, workGm, syncGm);
    for (int32_t i = 0; i < 256 * param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
