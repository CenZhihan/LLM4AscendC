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
#include <mockcpp/mockcpp.hpp>
#include "kernel_operator.h"
#include "common.h"
#include "kernel_utils.h"

using namespace std;
using namespace AscendC;

static const int tensorSize = 512;

namespace AscendC {
template <typename T>
class KernelAtomicDataCopy {
public:
    __aicore__ inline KernelAtomicDataCopy() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, int32_t pipeType, AtomicType atomicOpType)
    {
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        mPipeType = pipeType;
        if (mPipeType == 0) {
            pipe.InitBuffer(inQueueSrcVecIn, 1, tensorSize * sizeof(T));
        } else {
            pipe.InitBuffer(inQueueSrcL1In, 1, tensorSize * sizeof(T));
        }

        mAtomicType = atomicOpType;
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
        if (mPipeType == 0) {
            LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
            DataCopy(srcLocal, srcGlobal, tensorSize);
            inQueueSrcVecIn.EnQue(srcLocal);
        } else {
            LocalTensor<T> srcLocal = inQueueSrcL1In.AllocTensor<T>();
            DataCopy(srcLocal, srcGlobal, tensorSize);
            inQueueSrcL1In.EnQue(srcLocal);
        }
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        if (mPipeType == 0) {
            LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();
            if (mAtomicType == SUM) {
                SetAtomicAdd<T>();
            } else if (mAtomicType == MAX) {
                SetAtomicMax<T>();
            } else {
                SetAtomicMin<T>();
            }
            DataCopy(dstGlobal, dstLocal, tensorSize);
            SetAtomicNone();
            inQueueSrcVecIn.FreeTensor(dstLocal);
        } else {
            LocalTensor<T> dstLocal = inQueueSrcL1In.DeQue<T>();
            if (mAtomicType == SUM) {
                SetAtomicAdd<T>();
            } else if (mAtomicType == MAX) {
                SetAtomicMax<T>();
            } else {
                SetAtomicMin<T>();
            }
            DataCopy(dstGlobal, dstLocal, tensorSize);
            SetAtomicNone();
            inQueueSrcL1In.FreeTensor(dstLocal);
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::A1, 1> inQueueSrcL1In;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    AtomicType mAtomicType;
    int32_t mPipeType;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void AtomicDataCopy(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, int32_t pipeType, AtomicType atomicOpType)
{
    AscendC::KernelAtomicDataCopy<T> op;
    op.Init(dstGm, srcGm, pipeType, atomicOpType);
    op.Process();
}

struct AtomicDataCopyParams {
    int32_t typeSize;
    AtomicType atomicOpType;
    //pipeType = 0 for UB, pipeType = 1 for L1
    int32_t pipeType;
    void (*CalFunc)(uint8_t*, uint8_t*, int32_t, AtomicType);
};

class AtomicDataCopyTestsuite : public testing::Test,
    public testing::WithParamInterface<AtomicDataCopyParams> {
protected:
    void SetUp() {}
    void TearDown() {
        AscendC::CheckSyncState();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ATOMIC_DATACOPY, AtomicDataCopyTestsuite,
    ::testing::Values(AtomicDataCopyParams { sizeof(half), SUM, 0, AtomicDataCopy<half>},
        AtomicDataCopyParams { sizeof(float), SUM, 0, AtomicDataCopy<float>},
        AtomicDataCopyParams { sizeof(int8_t), SUM, 0, AtomicDataCopy<int8_t>},
        AtomicDataCopyParams { sizeof(int16_t), SUM, 0, AtomicDataCopy<int16_t>},
        AtomicDataCopyParams { sizeof(int32_t), SUM, 0, AtomicDataCopy<int32_t>},
        AtomicDataCopyParams { sizeof(bfloat16_t), SUM, 0, AtomicDataCopy<bfloat16_t>},
        AtomicDataCopyParams { sizeof(half), SUM, 1, AtomicDataCopy<half>},
        AtomicDataCopyParams { sizeof(float), SUM, 1, AtomicDataCopy<float>},
        AtomicDataCopyParams { sizeof(int8_t), SUM, 1, AtomicDataCopy<int8_t>},
        AtomicDataCopyParams { sizeof(int16_t), SUM, 1, AtomicDataCopy<int16_t>},
        AtomicDataCopyParams { sizeof(int32_t), SUM, 1, AtomicDataCopy<int32_t>},
        AtomicDataCopyParams { sizeof(bfloat16_t), SUM, 1, AtomicDataCopy<bfloat16_t>},
        AtomicDataCopyParams { sizeof(half), MAX, 0, AtomicDataCopy<half>},
        AtomicDataCopyParams { sizeof(float), MAX, 0, AtomicDataCopy<float>},
        AtomicDataCopyParams { sizeof(int8_t), MAX, 0, AtomicDataCopy<int8_t>},
        AtomicDataCopyParams { sizeof(int16_t), MAX, 0, AtomicDataCopy<int16_t>},
        AtomicDataCopyParams { sizeof(int32_t), MAX, 0, AtomicDataCopy<int32_t>},
        AtomicDataCopyParams { sizeof(bfloat16_t), MAX, 0, AtomicDataCopy<bfloat16_t>},
        AtomicDataCopyParams { sizeof(half), MIN, 0, AtomicDataCopy<half>},
        AtomicDataCopyParams { sizeof(float), MIN, 0, AtomicDataCopy<float>},
        AtomicDataCopyParams { sizeof(int8_t), MIN, 0, AtomicDataCopy<int8_t>},
        AtomicDataCopyParams { sizeof(int16_t), MIN, 0, AtomicDataCopy<int16_t>},
        AtomicDataCopyParams { sizeof(int32_t), MIN, 0, AtomicDataCopy<int32_t>},
        AtomicDataCopyParams { sizeof(bfloat16_t), MIN, 0, AtomicDataCopy<bfloat16_t>} ));

TEST_P(AtomicDataCopyTestsuite, AtomicDataCopyTestsuiteOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[tensorSize * param.typeSize] = {0};
    uint8_t dstGm[tensorSize * param.typeSize] = {0};

    param.CalFunc(dstGm, srcGm, param.pipeType, param.atomicOpType);
    for (int32_t i = 0; i < tensorSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

class TEST_ATOMIC_TYPE : public testing::Test {
protected:
    void SetUp()
    {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ATOMIC_TYPE, ATOMIC_TYPE)
{
    MOCKER(SetAtomicType<float>).expects(once());
    MOCKER(SetAtomicType<half>).expects(once());
    MOCKER(SetAtomicType<int16_t>).expects(once());
    MOCKER(SetAtomicType<int32_t>).expects(once());
    MOCKER(SetAtomicType<int8_t>).expects(once());
    MOCKER(SetAtomicType<bfloat16_t>).expects(once());
    SetAtomicType<float>();
    SetAtomicType<half>();
    SetAtomicType<int16_t>();
    SetAtomicType<int32_t>();
    SetAtomicType<int8_t>();
    SetAtomicType<bfloat16_t>();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}