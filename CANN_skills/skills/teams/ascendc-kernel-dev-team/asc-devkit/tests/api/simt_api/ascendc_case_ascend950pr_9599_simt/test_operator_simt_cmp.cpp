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
using namespace AscendC::Simt;
#define THREAD_DIM 128
template <typename T>
class KernelCmp {
    public:
        __aicore__ KernelCmp() {}
        __aicore__ inline void Process(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelCmpCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    int offset;
    for (int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx<128;idx+=block_num*GetThreadNum<0>()) {
        if (mode == 0) {
            if (idx < 64) {
                dst[idx] = IsFinite(src0[idx]);
                dst[idx+64] = IsNan(src0[idx+64]);
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelCmp<T>::Process(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    AscendC::Simt::VF_CALL<KernelCmpCompute<T>>(Dim3(THREAD_DIM, 1, 1), dst, src0, src1, mode);
}


struct CmpParams {
    int32_t mode;
};

class CmpTestsuite : public testing::Test, public testing::WithParamInterface<CmpParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};


INSTANTIATE_TEST_CASE_P(CmpTestCase, CmpTestsuite,
    ::testing::Values(CmpParams {0}));

TEST_P(CmpTestsuite, CmpTestCase)
{
    auto param = GetParam();
    int32_t mode = mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    uint8_t srcGm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};
    KernelCmp<float> op;
    op.Process((__gm__ float*)dstGm, (__gm__ float*)srcGm, (__gm__ float*)src1Gm, mode);
}
