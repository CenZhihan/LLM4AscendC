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
#include <cmath>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128

template <typename T, int32_t Mode>
class KernelSimtIntCompute {
    public:
        __aicore__ KernelSimtIntCompute() {}
        __aicore__ inline void Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2);
};

template <typename T, int32_t Mode>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__  void KernelSimtIntComputeCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2)
{
    for(int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx < 128; idx+=block_num*GetThreadNum<0>()) {
        if constexpr (Mode == 1) {
            dst[idx] = Clz(src0[idx]);
        } else if constexpr (Mode == 2) {
            dst[idx] = Popc(src0[idx]);
        } else if constexpr (Mode == 3) {
            dst[idx] = MulHi(src0[idx], src1[idx]);
        } else if constexpr (Mode == 4) {
            dst[idx] = BytePerm(src0[idx], src1[idx], src2[idx]);
        }
    }
}

template <typename T, int32_t Mode>
__aicore__ inline void KernelSimtIntCompute<T, Mode>::Process(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2)
{
    AscendC::Simt::VF_CALL<KernelSimtIntComputeCompute<T, Mode>>(Dim3(THREAD_DIM, 1, 1), dst, src0, src1, src2);
}

struct SimtIntComputeParams {
    void (*CallFunc)();
};

template<typename T, int32_t Mode>
void SimtIntComputeRunCase() {
    int byteSize = 4;
    int shape_size = 128;

    uint8_t dstGm[shape_size * byteSize] = {0};
    uint8_t src0Gm[shape_size * byteSize] = {0};
    uint8_t src1Gm[shape_size * byteSize] = {0};
    uint8_t src2Gm[shape_size * byteSize] = {0};
    KernelSimtIntCompute<T, Mode> op;
    op.Process((__gm__ T*)dstGm, (__gm__ T*)src0Gm, (__gm__ T*)src1Gm, (__gm__ T*)src2Gm);
}

class SimtIntComputeTestsuite : public testing::Test, public testing::WithParamInterface<SimtIntComputeParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(SimtIntComputeTestCase, SimtIntComputeTestsuite,
    ::testing::Values(
    SimtIntComputeParams {SimtIntComputeRunCase<uint32_t, 1>},
    SimtIntComputeParams {SimtIntComputeRunCase<uint32_t, 2>},
    SimtIntComputeParams {SimtIntComputeRunCase<uint32_t, 3>},
    SimtIntComputeParams {SimtIntComputeRunCase<uint32_t, 4>} // BytePerm
    ));

TEST_P(SimtIntComputeTestsuite, SimtIntComputeTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}

TEST_F(SimtIntComputeTestsuite, SimtIntBrev)
{
    uint32_t x = 123;
    uint64_t y = 123;
    EXPECT_EQ(3724541952, Brev(x));
    EXPECT_EQ(15996785876420001792, Brev(y));
}

TEST_F(SimtIntComputeTestsuite, SimtIntFfs)
{
    int64_t x = 123;
    int32_t y = 3724541952;
    EXPECT_EQ(1, Ffs(x));
    EXPECT_EQ(26, Ffs(y));
}
