/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEST_API_CHECK_REDUCE_REDUCE_CASE_COMMON_H
#define TEST_API_CHECK_REDUCE_REDUCE_CASE_COMMON_H

#include "include/adv_api/reduce/mean_utils.h"

using namespace AscendC;

template <typename T, class Params>
uint32_t ComputeTmpBufSize(Params &meanParams) 
{
    uint32_t elemNumPerRep = 64;
    if constexpr (sizeof(T) == sizeof(half)) {
        elemNumPerRep = 128;
    }
    uint32_t repeat = (meanParams.n + elemNumPerRep - 1) / elemNumPerRep;
    uint32_t finalWorkSize = meanParams.inner * sizeof(T) + (repeat + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    return finalWorkSize;
}


template <typename T>
uint32_t GetPadLast(const uint32_t srcShape[]) {
    uint32_t last = srcShape[1];
    uint32_t first = srcShape[0];
    constexpr uint32_t elePerBlk = ONE_BLK_SIZE / sizeof(T);
    uint32_t padLast = AlignUp(last, elePerBlk);
    return padLast;
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcShapeLastAxis(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = false;
    uint32_t srcShape[] = { 0, 32 };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcShapeArFirstAxis(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, first * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { 64, last };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcShapeRaLastAxis(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, 64 };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcShapeSrcSize(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { 64, 32 };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcInnerPad(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, last };
    bool srcInnerPad = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceSrcPos(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::A1, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, last };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceDstPos(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::A1, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, last };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceTmpPos(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::A1> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, last };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

template <typename T, void (*func)(
    const char*, const AscendC::LocalTensor<T>&, const AscendC::LocalTensor<T>&,
    const AscendC::LocalTensor<uint8_t>&, const uint32_t*, bool, uint32_t)>
void inline CheckReduceOverlap(__gm__ const char* apiName)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t first = 32;
    uint32_t last = 32;
    pipe.InitBuffer(inQueueX, 1, first * last * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, last * sizeof(T));
    pipe.InitBuffer(tmplocalBuf, last);
    AscendC::LocalTensor<T> srcTensor = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> dstTensor = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = true;
    uint32_t srcShape[] = { first, last };
    bool srcInnerPad = true;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    func(apiName, srcTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, GetPadLast<T>(srcShape));
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}


#endif // TEST_API_CHECK_REDUCE_REDUCE_CASE_COMMON_H
