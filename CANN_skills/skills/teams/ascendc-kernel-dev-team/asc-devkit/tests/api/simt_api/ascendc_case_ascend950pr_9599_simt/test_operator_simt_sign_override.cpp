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

#define D_TYPE int64_t
#define M 3
#define THREAD_DIM 128

template <typename T>
class KernelSignOverride {
    public:
        __aicore__  KernelSignOverride() {}
        __aicore__  inline void Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__  void KernelSignOverrideCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    for(int idx = GetThreadIdx<0>() + block_idx*GetThreadNum<0>(); idx < 128; idx+=block_num*GetThreadNum<0>())
    {
        if (mode == 0) {
            dst[idx] = src0[idx] + src1[idx];
        } else if (mode == 1) {
            dst[idx] = src0[idx] - src1[idx];
        } else if (mode == 2) {
            dst[idx] = src0[idx] * src1[idx];
        } else if (mode == 3) {
            dst[idx] = src0[idx] / (src1[idx] + 1);
        } else if (mode == 4) {
            dst[idx] = src0[idx] % (src1[idx] + 1);
        } else if (mode == 5) {
            dst[idx] = src0[idx];
            dst[idx]++;
        } else if (mode == 6) {
            dst[idx] = src0[idx];
            dst[idx]--;
        } else if (mode == 7) {
            dst[idx] = ++src0[idx];
        } else if (mode == 8) {
            dst[idx] = --src0[idx];
        } else if (mode == 9) {
            dst[idx] = src0[idx] >= src1[idx];
        } else if (mode == 10) {
            dst[idx] = src0[idx] <= src1[idx];
        } else if (mode == 11) {
            dst[idx] = src0[idx] > src1[idx];
        } else if (mode == 12) {
            dst[idx] = src0[idx] < src1[idx];
        } else if (mode == 13) {
            dst[idx] = src0[idx] == src1[idx];
        } else if (mode == 14) {
            dst[idx] = src0[idx] != src1[idx];
        } else if (mode == 15) {
            dst[idx] = (src0[idx] > 0) && (src1[idx] > 0);
        } else if (mode == 16) {
            dst[idx] = (src0[idx] > 0) || (src1[idx] > 0);
        } else if (mode == 17) {
            dst[idx] = !(src0[idx] > 0);
        } else if (mode == 18) {
            dst[idx] = src0[idx] > 0 ? 1:-1;
        } else if (mode == 19) {
            dst[idx] = src0[idx] & src1[idx];
        } else if (mode == 20) {
            dst[idx] = src0[idx] | src1[idx];
        } else if (mode == 21) {
            dst[idx] = src0[idx] ^ src1[idx];
        } else if (mode == 22) {
            dst[idx] = ~src0[idx];
        } else if (mode == 23) {
            dst[idx] = src0[idx] >> src1[idx];
        } else if (mode == 24) {
            dst[idx] = src0[idx] << src1[idx];
        }
    }
}

template <typename T>
__aicore__  inline void KernelSignOverride<T>::Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    AscendC::Simt::VF_CALL<KernelSignOverrideCompute<T>>(Dim3(THREAD_DIM, 1, 1), out, src0, src1, mode);
}

struct SignOverrideParams {
    int32_t mode;
};

class SignOverrideTestsuite : public testing::Test, public testing::WithParamInterface<SignOverrideParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};


INSTANTIATE_TEST_CASE_P(SignOverrideTestCase, SignOverrideTestsuite,
    ::testing::Values(SignOverrideParams {0},
    SignOverrideParams {1},
    SignOverrideParams {2},
    SignOverrideParams {3},
    SignOverrideParams {4},
    SignOverrideParams {5},
    SignOverrideParams {6},
    SignOverrideParams {7},
    SignOverrideParams {8},
    SignOverrideParams {9},
    SignOverrideParams {10},
    SignOverrideParams {11},
    SignOverrideParams {12},
    SignOverrideParams {13},
    SignOverrideParams {14},
    SignOverrideParams {15},
    SignOverrideParams {16},
    SignOverrideParams {17},
    SignOverrideParams {18},
    SignOverrideParams {19},
    SignOverrideParams {20},
    SignOverrideParams {21},
    SignOverrideParams {22},
    SignOverrideParams {23},
    SignOverrideParams {24}
                      ));

TEST_P(SignOverrideTestsuite, SignOverrideTestCase)
{
    auto param = GetParam();
    int32_t mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    uint8_t src0Gm[shapeSize * fpByteSize] = {0};
    uint8_t src1Gm[shapeSize * fpByteSize] = {0};
    KernelSignOverride<int32_t> op;
    op.Process((__gm__ int32_t*)dstGm, (__gm__ int32_t*)src0Gm, (__gm__ int32_t*)src1Gm, mode);
}
