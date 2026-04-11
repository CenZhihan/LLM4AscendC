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

#define DType float
#define THREAD_DIM 128
template <typename T>
class KernelThreadBarrier {
    public:
        __aicore__ KernelThreadBarrier() {}
        __aicore__ inline void Process(__gm__ T* out);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelThreadBarrierCompute(__gm__ T* dst)
{
    for(int idx = GetThreadIdx<0>() + block_idx*GetThreadNum<0>(); idx < 256; idx+=block_num*GetThreadNum<0>())
    {
        if(idx > 0 && idx != 128) {
            dst[idx] = 1;
        }

        ThreadBarrier();
        // 测试核内是否同步
        if (idx==0) {
            dst[0] = 0;
            for(int i=127; i>0;i--) {
                dst[0] += dst[i];
            }
        }

        ThreadBarrier();
        if(idx > 0 && idx != 128) {
            dst[idx] = -1;
        }
    }
}

template <typename T>
__aicore__ inline void KernelThreadBarrier<T>::Process(__gm__ T* dst)
{
    AscendC::Simt::VF_CALL<KernelThreadBarrierCompute<T>>(Dim3(THREAD_DIM, 1, 1), dst);
}

struct ThreadBarrierParams {
    int32_t mode;
};

class ThreadBarrierTestsuite : public testing::Test, public testing::WithParamInterface<ThreadBarrierParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(ThreadBarrierTestCase, ThreadBarrierTestsuite,
    ::testing::Values(ThreadBarrierParams {0}
                      ));

TEST_P(ThreadBarrierTestsuite, ThreadBarrierTestCase)
{
    auto param = GetParam();
    int fp_byte_size = 4;
    int shape_size = 256;

    uint8_t dstGm[shape_size * fp_byte_size] = {0};
    KernelThreadBarrier<float> op;
    op.Process((__gm__ float*)dstGm);
}
