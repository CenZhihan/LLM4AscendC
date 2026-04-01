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

using namespace AscendC;

namespace AscendC {
template <typename srcType>
class KernelWhere {
public:
    __aicore__ inline KernelWhere()
    {}
    __aicore__ inline void Init(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR condition_gm, GM_ADDR dst_gm,
                                const uint32_t count, const uint32_t mode)
    {
        this->mode = mode;
        dataSize = count;
        src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0_gm), dataSize);
        src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1_gm), dataSize);
        condition_global.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(condition_gm), dataSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst_gm), dataSize);

        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueZ, 1, dataSize * sizeof(bool));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
    }
    __aicore__ inline void Process()
    {
        AscendCUtils::SetOverflow(1);
        CopyIn();
        Compute();
        CopyOut();
        AscendCUtils::SetOverflow(0);
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<srcType> src0Local = inQueueX.AllocTensor<srcType>();
        LocalTensor<srcType> src1Local = inQueueY.AllocTensor<srcType>();
        LocalTensor<bool> conditionLocal = inQueueZ.AllocTensor<bool>();
        DataCopy(src0Local, src0_global, dataSize);
        DataCopy(src1Local, src1_global, dataSize);
        DataCopy(conditionLocal, condition_global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueY.EnQue(src1Local);
        inQueueZ.EnQue(conditionLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();

        LocalTensor<srcType> src0Local = inQueueX.DeQue<srcType>();
        LocalTensor<srcType> src1Local = inQueueY.DeQue<srcType>();
        LocalTensor<bool> conditionLocal = inQueueZ.DeQue<bool>();
        const uint32_t mode0 = 0;
        const uint32_t mode1 = 2;
        const uint32_t mode2 = 2;

        if (mode = mode0) {
            Where(dstLocal, src0Local, src1Local, conditionLocal, dataSize);
        } else if (mode == mode1) {
            srcType src0 = src0Local.GetValue(0);
            Where(dstLocal, src0, src1Local, conditionLocal, dataSize);
        } else if (mode == mode2) {
            srcType src1 = src1Local.GetValue(0);
            Where(dstLocal, src0Local, src1, conditionLocal, dataSize);
        } else {
            srcType src0 = src0Local.GetValue(0);
            srcType src1 = src1Local.GetValue(0);
            Where(dstLocal, src0, src1, conditionLocal, dataSize);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(src0Local);
        inQueueY.FreeTensor(src1Local);
        inQueueZ.FreeTensor(conditionLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        DataCopy(dst_global, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src0_global;
    GlobalTensor<srcType> src1_global;
    GlobalTensor<bool> condition_global;
    GlobalTensor<srcType> dst_global;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECIN, 1> inQueueZ;
    TQue<QuePosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
    uint32_t mode = 0;
};
}

template <typename srcType>
__aicore__ void kernel_where_operator(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR condition_gm, GM_ADDR dst_gm,
                                const uint32_t count, const uint32_t mode)
{
    KernelWhere<srcType> op;
    op.Init(src0_gm, src1_gm, condition_gm, dst_gm, count, mode);
    op.Process();
}

struct WhereTestParams {
    uint32_t dataTypeSize;
    uint32_t inDataSize;
    uint32_t inMode;
    void (*calFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t);
};

class AdvanceWhereTestSuite : public testing::TestWithParam<WhereTestParams> {
protected:
void SetUp() {}
void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_WHERE, AdvanceWhereTestSuite,
    ::testing::Values(
        WhereTestParams {4, 32, 0, kernel_where_operator<float>},
        WhereTestParams {4, 4096, 1, kernel_where_operator<float>},
        WhereTestParams {4, 64, 2, kernel_where_operator<float>},
        WhereTestParams {4, 96, 3, kernel_where_operator<float>},
        WhereTestParams {2, 32, 0, kernel_where_operator<half>},
        WhereTestParams {2, 4096, 1, kernel_where_operator<half>},
        WhereTestParams {2, 96, 2, kernel_where_operator<half>},
        WhereTestParams {2, 64, 3, kernel_where_operator<half>},
        WhereTestParams {1, 32, 0, kernel_where_operator<int8_t>},
        WhereTestParams {1, 4096, 1, kernel_where_operator<int8_t>},
        WhereTestParams {1, 256, 2, kernel_where_operator<int8_t>},
        WhereTestParams {1, 1024, 3, kernel_where_operator<int8_t>},
        WhereTestParams {8, 32, 0, kernel_where_operator<int64_t>},
        WhereTestParams {8, 1024, 1, kernel_where_operator<int64_t>},
        WhereTestParams {8, 96, 2, kernel_where_operator<int64_t>},
        WhereTestParams {8, 64, 3, kernel_where_operator<int64_t>}
    )
);

TEST_P(AdvanceWhereTestSuite, kernel_where_operator)
{
    auto param = GetParam();
    uint32_t dataTypeSize = param.dataTypeSize;
    uint32_t inDataSize = param.inDataSize;
    uint32_t inMode = param.inMode;
    uint8_t input2Gm[inDataSize * dataTypeSize] = {0};
    uint8_t input1Gm[inDataSize * dataTypeSize] = {0};
    uint8_t input0Gm[inDataSize * dataTypeSize] = {0};
    uint8_t outputGm[inDataSize * dataTypeSize] = {0};
    param.calFunc(input0Gm, input1Gm, input2Gm, outputGm, inDataSize, inMode);
    for (int32_t i = 0; i < inDataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
