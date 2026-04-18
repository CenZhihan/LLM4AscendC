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
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
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
                // dst[idx] = __hisfinite(src0[idx]);
                dst[idx+64] = __hisnan(src0[idx+64]);
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelCmp<T>::Process(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    asc_vf_call<KernelCmpCompute<T>>(Dim3(THREAD_DIM, 1, 1), dst, src0, src1, mode);
}

// ================================ Test half start ================================
struct CmpParams_half {
    int32_t mode;
};

class CmpTestsuite_half : public testing::Test, public testing::WithParamInterface<CmpParams_half> {
protected:
    void SetUp() {}
    void TearDown() {}
};


INSTANTIATE_TEST_CASE_P(CmpTestCase_half, CmpTestsuite_half,
    ::testing::Values(CmpParams_half {0}));

TEST_P(CmpTestsuite_half, CmpTestCase_half)
{
    auto param = GetParam();
    int32_t mode = mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    uint8_t srcGm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};
    KernelCmp<half> op;
    op.Process((__gm__ half*)dstGm, (__gm__ half*)srcGm, (__gm__ half*)src1Gm, mode);
}
// ================================ Test half end ==================================

// ================================ Test bfloat16_t start ================================
struct CmpParams_bfloat16t {
    int32_t mode;
};

class CmpTestsuite_bfloat16t : public testing::Test, public testing::WithParamInterface<CmpParams_bfloat16t> {
protected:
    void SetUp() {}
    void TearDown() {}
};


INSTANTIATE_TEST_CASE_P(CmpTestCase_bfloat16t, CmpTestsuite_bfloat16t,
    ::testing::Values(CmpParams_bfloat16t {0}));

TEST_P(CmpTestsuite_bfloat16t, CmpTestCase_bfloat16t)
{
    auto param = GetParam();
    int32_t mode = mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    uint8_t srcGm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};
    KernelCmp<bfloat16_t> op;
    op.Process((__gm__ bfloat16_t*)dstGm, (__gm__ bfloat16_t*)srcGm, (__gm__ bfloat16_t*)src1Gm, mode);
}
// ================================ Test bfloat16_t end ==================================