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
using AscendC::Reg::RegTensor;
using AscendC::Reg::MaskReg;
using AscendC::Reg::UpdateMask;

template <typename T, uint8_t mode>
class KernelAtomicOp {
public:
    __aicore__ inline KernelAtomicOp()
    {}

    __aicore__ inline void Process(GM_ADDR dst0Gm, GM_ADDR src0Gm, uint32_t calcount)
    {
        dst0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst0Gm), calcount);
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src0Gm), calcount);
        pipe.InitBuffer(inQueueX, 1, calcount * sizeof(T));
        PipeBarrier<PIPE_ALL>();
        T scalar = 1;
 
        dataSize = calcount;
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> src0Local = inQueueX.AllocTensor<T>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(src0Local, src0Global, dataSize);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dst0Global, src0Local, dataSize);
        PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
 
        if constexpr(mode == 0) {
            T a = AtomicAdd(reinterpret_cast<__gm__ T *>(dst0Gm) + GetBlockIdx() * 2, scalar);
            T b = AtomicAdd(reinterpret_cast<__gm__ T *>(dst0Gm) + 1 + GetBlockIdx() * 2, a);
        } else if constexpr(mode == 1) {
            T a = AtomicMax(reinterpret_cast<__gm__ T *>(dst0Gm) + GetBlockIdx() * 2, scalar);
            T b = AtomicMax(reinterpret_cast<__gm__ T *>(dst0Gm) + 1 + GetBlockIdx() * 2, a);
        } else if constexpr(mode == 2) {
            T a = AtomicMin(reinterpret_cast<__gm__ T *>(dst0Gm) + GetBlockIdx() * 2, scalar);
            T b = AtomicMin(reinterpret_cast<__gm__ T *>(dst0Gm) + 1 + GetBlockIdx() * 2, a);
        } else if constexpr(mode == 3) {
            T a = AtomicCas(reinterpret_cast<__gm__ T *>(dst0Gm) + GetBlockIdx() * 2, dst0Global.GetValue(GetBlockIdx() * 2), static_cast<T>(2));
            T b = AtomicExch(reinterpret_cast<__gm__ T *>(dst0Gm) + 1 + GetBlockIdx() * 2, a);
        } else if constexpr(mode == 4) {
            T a = AtomicExch(reinterpret_cast<__gm__ T *>(dst0Gm) + GetBlockIdx() * 2, scalar);
            T b = AtomicExch(reinterpret_cast<__gm__ T *>(dst0Gm) + 1 + GetBlockIdx() * 2, a);
        } else if constexpr(mode == 5) {
            // multicore inplace operation
            T a = AtomicAdd(reinterpret_cast<__gm__ T *>(dst0Gm), scalar);
        } else if constexpr(mode == 6) {
            // multicore inplace operation
            T a = AtomicMax(reinterpret_cast<__gm__ T *>(dst0Gm), scalar);
        } else if constexpr(mode == 7) {
            // multicore inplace operation
            T a = AtomicMin(reinterpret_cast<__gm__ T *>(dst0Gm), scalar);
        }
        PipeBarrier<PIPE_ALL>();
    }
 
private:
    GlobalTensor<T> src0Global;
    GlobalTensor<T> dst0Global;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    uint32_t dataSize = 0;
};
 
struct MicroAtomicOpParams {
    void (*CallFunc)();
};
 
template<typename T, uint8_t mode>
void RunCase() {
    int srcByteSize = sizeof(T);
    int dstByteSize = sizeof(T);
    int dataSize = 256;
    uint8_t dstGm0[dataSize * dstByteSize] = {0};
    uint8_t srcGm0[dataSize * srcByteSize] = {0};
 
    KernelAtomicOp<T, mode> op;
    op.Process(dstGm0, srcGm0, dataSize);
}

class MicroAtomicOpTestSuite : public testing::Test, public testing::WithParamInterface<MicroAtomicOpParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroAtomicOpTestCases, MicroAtomicOpTestSuite,
    ::testing::Values(MicroAtomicOpParams { RunCase<float, 0>},
                      MicroAtomicOpParams { RunCase<uint64_t, 0>},
                      MicroAtomicOpParams { RunCase<int64_t, 0>},
                      MicroAtomicOpParams { RunCase<uint32_t, 0>},
                      MicroAtomicOpParams { RunCase<int32_t, 0>},
                      MicroAtomicOpParams { RunCase<float, 1>},
                      MicroAtomicOpParams { RunCase<uint64_t, 1>},
                      MicroAtomicOpParams { RunCase<int64_t, 1>},
                      MicroAtomicOpParams { RunCase<uint32_t, 1>},
                      MicroAtomicOpParams { RunCase<int32_t, 1>},
                      MicroAtomicOpParams { RunCase<float, 2>},
                      MicroAtomicOpParams { RunCase<uint64_t, 2>},
                      MicroAtomicOpParams { RunCase<int64_t, 2>},
                      MicroAtomicOpParams { RunCase<uint32_t, 2>},
                      MicroAtomicOpParams { RunCase<int32_t, 2>},
                      MicroAtomicOpParams { RunCase<float, 5>},
                      MicroAtomicOpParams { RunCase<uint64_t, 5>},
                      MicroAtomicOpParams { RunCase<int64_t, 5>},
                      MicroAtomicOpParams { RunCase<uint32_t, 5>},
                      MicroAtomicOpParams { RunCase<int32_t, 5>},
                      MicroAtomicOpParams { RunCase<float, 6>},
                      MicroAtomicOpParams { RunCase<uint64_t, 6>},
                      MicroAtomicOpParams { RunCase<int64_t, 6>},
                      MicroAtomicOpParams { RunCase<uint32_t, 6>},
                      MicroAtomicOpParams { RunCase<int32_t, 6>},
                      MicroAtomicOpParams { RunCase<float, 7>},
                      MicroAtomicOpParams { RunCase<uint64_t, 7>},
                      MicroAtomicOpParams { RunCase<int64_t, 7>},
                      MicroAtomicOpParams { RunCase<uint32_t, 7>},
                      MicroAtomicOpParams { RunCase<int32_t, 7>},
                      MicroAtomicOpParams { RunCase<uint32_t, 3>},
                      MicroAtomicOpParams { RunCase<uint64_t, 3>},
                      MicroAtomicOpParams { RunCase<uint32_t, 4>},
                      MicroAtomicOpParams { RunCase<uint64_t, 4>}
                      ));
 
TEST_P(MicroAtomicOpTestSuite, MicroAtomicOpTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}