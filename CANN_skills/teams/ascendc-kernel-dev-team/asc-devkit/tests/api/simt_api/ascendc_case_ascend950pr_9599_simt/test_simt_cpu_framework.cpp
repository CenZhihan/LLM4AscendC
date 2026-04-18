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
#include <mutex>
#include "kernel_operator.h"
#include "mockcpp/mockcpp.hpp"

using namespace AscendC;
using namespace AscendC::Simt;

enum class TestMode {
    NORMAL,
    REDUCE,
    ATOMIC,
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ inline void SimtCompute(const LocalTensor<T> &ub, GlobalTensor<T> &srcGm, GlobalTensor<T> &dstGm, uint32_t srcSize_, uint32_t groupSize_)
    {
        for (uint32_t idx = AscendC::Simt::GetThreadIdx<0>(); idx < srcSize_; idx += AscendC::Simt::GetThreadNum<0>()) {
            ub(idx) = srcGm(idx);
            dstGm(idx % groupSize_) += ub(idx);
            AscendC::Simt::ThreadBarrier();
        }
    }

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ inline void SimtComputeReduce(const LocalTensor<T> &ub, GlobalTensor<T> &srcGm, GlobalTensor<T> &dstGm, uint32_t srcSize_, uint32_t groupSize_)
{
    for (uint32_t idx = AscendC::Simt::GetThreadIdx<0>(); idx < srcSize_; idx += AscendC::Simt::GetThreadNum<0>()) {
        T sum = AscendC::Simt::WarpReduceAddSync(srcGm(idx));
        dstGm(idx % groupSize_) = sum;
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) __aicore__ inline void SimtComputeAtomic(const LocalTensor<T> &ub, GlobalTensor<T> &srcGm, GlobalTensor<T> &dstGm, uint32_t srcSize_, uint32_t groupSize_)
{
    for (uint32_t idx = AscendC::Simt::GetThreadIdx<0>(); idx < srcSize_; idx += AscendC::Simt::GetThreadNum<0>()) {
        AscendC::Simt::AtomicMax(dstGm.GetPhyAddr(idx % groupSize_), srcGm(idx));
    }
}

template <typename T, TestMode mode> class KernelSimtCpu {
public:
    __aicore__ inline KernelSimtCpu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t srcSize, uint32_t groupSize, TPipe *pipeIn)
    {
        pipe = pipeIn;
        groupSize1 = groupSize;
        srcSize1 = srcSize;

        srcGm.SetGlobalBuffer((__gm__ T *)x, srcSize1);
        dstGm.SetGlobalBuffer((__gm__ T *)z, groupSize1);
        // pipe alloc memory to queue, the unit is Bytes
        pipe->InitBuffer(inQueue, 1, srcSize1 * sizeof(T));
    }

    __aicore__ inline void Process(uint32_t threadNum)
    {
        SimtProcess(threadNum);
    }

private:
    __aicore__ inline void SimtProcess(uint32_t threadNum)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        if constexpr (mode == TestMode::NORMAL) {
            AscendC::Simt::VF_CALL<SimtCompute<T>>(Simt::Dim3{threadNum}, srcLocal, srcGm, dstGm, srcSize1, groupSize1);
        }
        if constexpr (mode == TestMode::REDUCE) {
            AscendC::Simt::VF_CALL<SimtComputeReduce<T>>(Simt::Dim3{threadNum}, srcLocal, srcGm, dstGm, srcSize1, groupSize1);
        } else if constexpr (mode == TestMode::ATOMIC) {
            AscendC::Simt::VF_CALL<SimtComputeAtomic<T>>(Simt::Dim3{threadNum}, srcLocal, srcGm, dstGm, srcSize1, groupSize1);
        }
    }

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> inQueue;
    GlobalTensor<T> srcGm;
    GlobalTensor<T> dstGm;
    uint32_t srcSize1 = 0;
    uint32_t groupSize1 = 0;
};

class SimtCpuTestSuite : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }
};

__simt_vf__ __aicore__ inline void SimtComputeWarp(uint32_t& warpSum, uint32_t& threadSum, uint32_t& laneSum)
{
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    warpSum += AscendC::Simt::GetWarpId();
    threadSum += AscendC::Simt::GetThreadIdx();
    laneSum += AscendC::Simt::GetLaneId();
}

TEST_F(SimtCpuTestSuite, basicFunctionMarco)
{
    constexpr uint32_t threadNum = 512;
    constexpr uint32_t warpProc = 16;
    uint32_t warpSum = 0;
    uint32_t threadSum = 0;
    uint32_t laneSum = 0;
    constexpr uint32_t expectedWarpSum = (threadNum / AscendC::Simt::THREAD_PER_WARP) * (threadNum / AscendC::Simt::THREAD_PER_WARP - 1) * warpProc;
    constexpr uint32_t expectedThreadSum = (threadNum - 1) * threadNum / 2;
    constexpr uint32_t expectedLaneSum = (AscendC::Simt::THREAD_PER_WARP - 1) * threadNum / 2;
    cce::async_invoke<SimtComputeWarp>(cce::dim3{ static_cast<uint32_t>(threadNum) }, warpSum, threadSum, laneSum);

    EXPECT_EQ(warpSum, expectedWarpSum);
    EXPECT_EQ(threadSum, expectedThreadSum);
    EXPECT_EQ(laneSum, expectedLaneSum);
}

TEST_F(SimtCpuTestSuite, basicFunctionFunc)
{
    constexpr uint32_t threadNum = 512;
    constexpr uint32_t warpProc = 16;
    uint32_t warpSum = 0;
    uint32_t threadSum = 0;
    uint32_t laneSum = 0;
    constexpr uint32_t expectedWarpSum = (threadNum / AscendC::Simt::THREAD_PER_WARP) * (threadNum / AscendC::Simt::THREAD_PER_WARP - 1) * warpProc;
    constexpr uint32_t expectedThreadSum = (threadNum - 1) * threadNum / 2;
    constexpr uint32_t expectedLaneSum = (AscendC::Simt::THREAD_PER_WARP - 1) * threadNum / 2;
    cce::async_invoke<SimtComputeWarp>(cce::dim3{ static_cast<uint32_t>(threadNum) }, warpSum, threadSum, laneSum);

    EXPECT_EQ(warpSum, expectedWarpSum);
    EXPECT_EQ(threadSum, expectedThreadSum);
    EXPECT_EQ(laneSum, expectedLaneSum);
}

#define SIMT_CPU_TEST_NORMAL(threadNum)                                       \
    TEST_F(SimtCpuTestSuite, normalWithThread##threadNum)                     \
    {                                                                         \
        TPipe pipe;                                                           \
        KernelSimtCpu<uint32_t, TestMode::NORMAL> op;                         \
        uint32_t srcSize = 4096;                                              \
        uint32_t dstSize = 1024;                                              \
        uint32_t srcGm[srcSize] = { 0 };                                      \
        for (uint32_t i = 0; i < srcSize; i++) {                              \
            srcGm[i] = i;                                                     \
        }                                                                     \
        uint32_t expectedResult[dstSize] = { 0 };                             \
        for (int i = 0; i < srcSize; i++) {                                   \
            expectedResult[i % dstSize] += srcGm[i];                          \
        }                                                                     \
        uint32_t dstGm[dstSize] = { 0 };                                      \
        op.Init((uint8_t *)srcGm, (uint8_t *)dstGm, srcSize, dstSize, &pipe); \
        op.Process(threadNum);                                                \
        for (int i = 0; i < dstSize; i++) {                                   \
            EXPECT_EQ(expectedResult[i], dstGm[i]);                           \
        }                                                                     \
    }

#define SIMT_CPU_TEST_REDUCE(threadNum)                                            \
    TEST_F(SimtCpuTestSuite, reduceWithThread##threadNum)                          \
    {                                                                              \
        TPipe pipe;                                                                \
        KernelSimtCpu<uint32_t, TestMode::REDUCE> op;                              \
        uint32_t srcSize = 1024;                                                   \
        uint32_t dstSize = 1024;                                                   \
        uint32_t srcGm[srcSize] = { 0 };                                           \
        for (uint32_t i = 0; i < srcSize; i++) {                                   \
            srcGm[i] = i;                                                          \
        }                                                                          \
        constexpr uint32_t threadPerWarp = 32;                                     \
        uint32_t expectedResult[dstSize] = { 0 };                                  \
        for (int i = 0; i < srcSize; i++) {                                        \
            expectedResult[i / threadPerWarp * threadPerWarp] += srcGm[i];         \
        }                                                                          \
        for (int i = 0; i < srcSize; i++) {                                        \
            expectedResult[i] = expectedResult[i / threadPerWarp * threadPerWarp]; \
        }                                                                          \
                                                                                   \
        uint32_t dstGm[dstSize] = { 0 };                                           \
        op.Init((uint8_t *)srcGm, (uint8_t *)dstGm, srcSize, dstSize, &pipe);      \
        op.Process(threadNum);                                                     \
                                                                                   \
        for (int i = 0; i < dstSize; i++) {                                        \
            EXPECT_EQ(expectedResult[i], dstGm[i]);                                \
        }                                                                          \
    }

#define SIMT_CPU_TEST_ATOMIC(threadNum)                                                    \
    TEST_F(SimtCpuTestSuite, atomicWithThread##threadNum)                                  \
    {                                                                                      \
        TPipe pipe;                                                                        \
        KernelSimtCpu<uint32_t, TestMode::ATOMIC> op;                                      \
        uint32_t srcSize = 4096;                                                           \
        uint32_t dstSize = 32;                                                             \
        uint32_t srcGm[srcSize] = { 0 };                                                   \
        for (uint32_t i = 0; i < srcSize; i++) {                                           \
            srcGm[i] = i;                                                                  \
        }                                                                                  \
        uint32_t expectedResult[dstSize] = { 0 };                                          \
        for (int i = 0; i < srcSize; i++) {                                                \
            expectedResult[i % dstSize] = std::max(expectedResult[i % dstSize], srcGm[i]); \
        }                                                                                  \
        uint32_t dstGm[dstSize] = { 0 };                                                   \
        op.Init((uint8_t *)srcGm, (uint8_t *)dstGm, srcSize, dstSize, &pipe);              \
        op.Process(threadNum);                                                             \
                                                                                           \
        for (int i = 0; i < dstSize; i++) {                                                \
            EXPECT_EQ(expectedResult[i], dstGm[i]);                                        \
        }                                                                                  \
    }


// SIMT_CPU_TEST_NORMAL(1024)
// SIMT_CPU_TEST_NORMAL(32)
// SIMT_CPU_TEST_NORMAL(2048)
// SIMT_CPU_TEST_REDUCE(1024)
// SIMT_CPU_TEST_REDUCE(32)
// SIMT_CPU_TEST_REDUCE(2048)
// SIMT_CPU_TEST_ATOMIC(1024)
// SIMT_CPU_TEST_ATOMIC(32)
// SIMT_CPU_TEST_ATOMIC(2048)
