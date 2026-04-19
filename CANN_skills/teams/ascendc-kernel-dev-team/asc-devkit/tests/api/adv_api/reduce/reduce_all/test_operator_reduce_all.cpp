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

#include <iostream>
#include <fstream>

namespace AscendC {

template <typename T, bool isReuse, bool isAr>
class KernelReduceAll {
public:
__aicore__ inline KernelReduceAll() {}
__aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t firstAxis, uint32_t lastAxis)
{

    first = firstAxis;
    last = lastAxis;
    uint32_t padLast = AlignUp(last, ONE_BLK_SIZE / sizeof(T));
    uint32_t reduceAxis = isAr ? last : first;
    srcSize = padLast * first;
    padRet = AlignUp(reduceAxis, ONE_BLK_SIZE / sizeof(T));
    uint32_t k = 0, firstCopy = first;
    while (firstCopy > 0) {
        k++;
        firstCopy >>= 1;
    }
    uint32_t splitK = 1 << (k - 1);
    constexpr uint32_t elePerBlk = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t elePerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize * sizeof(T));
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), padRet * sizeof(T));
    pipe.InitBuffer(inQueue, 1, srcSize * sizeof(T));
    pipe.InitBuffer(outQueue, 1, padRet * sizeof(T));

    if constexpr (!isReuse) {
        if constexpr (sizeof(T) == 4) {
            if constexpr (!isAr) {
                pipe.InitBuffer(tbuf, splitK * padLast * sizeof(T));
            } else {
                if (last < elePerRep) {
                    pipe.InitBuffer(tbuf, first * elePerBlk * sizeof(T));
                } else {
                    pipe.InitBuffer(tbuf, first * elePerRep * sizeof(T));
                }
            }
        } else {
            tmpSize = isAr ? ((first * ONE_BLK_SIZE + padLast * sizeof(uint16_t)) * sizeof(T)) : srcSize * sizeof(T);
            pipe.InitBuffer(tbuf, tmpSize);
        }
    }
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
    DataCopy(srcLocal, srcGlobal, srcSize);
    inQueue.EnQue(srcLocal);
}
__aicore__ inline void Compute()
{
    LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
    LocalTensor<T> srcLocal = inQueue.DeQue<T>();
    uint32_t shape[] = { first, last };
    if constexpr (!isReuse) {
        LocalTensor<uint8_t> tmp = tbuf.Get<uint8_t>();
        if constexpr (isAr) {
            ReduceAll<T, Pattern::Reduce::AR, isReuse>(dstLocal, srcLocal, tmp, shape, true);
        } else {
            ReduceAll<T, Pattern::Reduce::RA, isReuse>(dstLocal, srcLocal, tmp, shape, true);
        }
    } else {
        if constexpr (isAr) {
            ReduceAll<T, Pattern::Reduce::AR, isReuse>(dstLocal, srcLocal, shape, true);
        } else {
            ReduceAll<T, Pattern::Reduce::RA, isReuse>(dstLocal, srcLocal, shape, true);
        }
    }
    outQueue.EnQue<T>(dstLocal);
    inQueue.FreeTensor(srcLocal);
}
__aicore__ inline void CopyOut()
{
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();
    DataCopy(dstGlobal, dstLocal, padRet);
    outQueue.FreeTensor(dstLocal);
}

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<> tbuf;
    uint32_t srcSize = 0;
    uint32_t padRet = 0;
    uint32_t first = 0;
    uint32_t last = 0;
    uint32_t tmpSize = 0;
};
} // namespace AscendC

template <typename T, bool isReuse>
__global__ __aicore__ void MainReduceAll(
    __gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t first, uint32_t last, bool isAr)
{
    if (isAr) {
        AscendC::KernelReduceAll<T, isReuse, true> op;
        op.Init(srcGm, dstGm, first, last);
        op.Process();
    } else {
        AscendC::KernelReduceAll<T, isReuse, false> op;
        op.Init(srcGm, dstGm, first, last);
        op.Process();
    }
}

struct ReduceAllTestParams {
    int32_t typeSize;
    int32_t first;
    int32_t last;
    bool isAr;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, bool);
};

class ReduceAllTestsuite : public testing::Test, public testing::WithParamInterface<ReduceAllTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPERATTION_REDUCE_ALL, ReduceAllTestsuite,
    ::testing::Values(
        ReduceAllTestParams { 1, 8, 32, true, MainReduceAll<uint8_t, true> },
        ReduceAllTestParams { 1, 16, 128, true, MainReduceAll<uint8_t, false> },
        ReduceAllTestParams { 1, 1, 8, false, MainReduceAll<uint8_t, true> },
        ReduceAllTestParams { 1, 8, 1, false, MainReduceAll<uint8_t, false> },
        ReduceAllTestParams { 1, 16, 7, false, MainReduceAll<uint8_t, true> },
        ReduceAllTestParams { 1, 8, 7, true, MainReduceAll<uint8_t, false> },
        ReduceAllTestParams { 4, 8, 32, true, MainReduceAll<float, true> },
        ReduceAllTestParams { 4, 7, 35, true, MainReduceAll<float, true> },
        ReduceAllTestParams { 4, 16, 128, true, MainReduceAll<float, false> },
        ReduceAllTestParams { 4, 1, 8, false, MainReduceAll<float, true> },
        ReduceAllTestParams { 4, 8, 1, false, MainReduceAll<float, false> },
        ReduceAllTestParams { 4, 16, 7, false, MainReduceAll<float, true> },
        ReduceAllTestParams { 4, 4, 8, true, MainReduceAll<float, false> },
        ReduceAllTestParams { 4, 7, 1, true, MainReduceAll<float, false> },
        ReduceAllTestParams { 4, 4, 64, true, MainReduceAll<float, true> }
        ));

TEST_P(ReduceAllTestsuite, ReduceAllOpTestCase)
{
    auto param = GetParam();
    auto first = param.first;
    auto last = param.last;
    constexpr uint32_t BLK_SIZE = 32;
    auto padLast = (last * param.typeSize + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;
    uint8_t srcGm[first * padLast] = {0}; // external guarantee inner is 32B aligned
    uint32_t dstLen = param.isAr ? first : last;
    auto padDst = (dstLen * param.typeSize + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;
    uint8_t dstGm[padDst] = {0};
    param.cal_func(dstGm, srcGm, first, last, param.isAr);
    EXPECT_EQ(dstGm[0], 0);
}
