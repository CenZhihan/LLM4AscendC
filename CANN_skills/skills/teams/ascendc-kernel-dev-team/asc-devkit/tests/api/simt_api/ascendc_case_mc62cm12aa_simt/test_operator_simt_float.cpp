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

template <typename T>
class KernelFloatCompute {
    public:
        __aicore__ KernelFloatCompute() {}
        __aicore__ inline void Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__  void KernelFloatComputeCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2, const int mode)
{
    int quo;
    int32_t s32n;
    int64_t n;
    src0[0] = NAN;
    src1[0] = NAN;
    src0[2] = INFINITY;
    src1[3] = INFINITY;
    for(int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx < 128; idx+=block_num*GetThreadNum<0>()) {
        switch (mode) {
            case 6:
                dst[idx] = Fma(src0[idx], src1[idx], src2[idx]);
                break;
            case 7:
                dst[idx] = Abs(src0[idx]);
                break;
            case 8:
                dst[idx] = Max(src0[idx], src1[idx]);
                break;
            case 9:
                dst[idx] = Min(src0[idx], src1[idx]);
                break;
            case 10:
                dst[idx] = Fdim(src0[idx], src1[idx]);
                break;
            case 11:
                dst[idx] = Mod(src0[idx], src1[idx]);
                break;
            case 12:
                dst[idx] = Remainder(src0[idx], src1[idx]);
                break;
            case 13:
                if (idx < 64) {
                    // for coverage
                    if (idx == 0) {
                        src0[idx] = 0.449568;
                        src1[idx] = 0.000535485;
                    } else if (idx == 1) {
                        src0[idx] = -1.72376e+38;
                        src1[idx] = 2.23178e+37;
                    }
                    dst[idx] = RemQuo(src0[idx], src1[idx], &quo);
                    dst[idx+64] = quo;
                }
                break;
            case 14:
                dst[idx] = CopySign(src0[idx], src1[idx]);
                break;
            case 15:
                dst[idx] = NearbyInt(src0[idx]);
                break;
            case 16:
                dst[idx] = NextAfter(src0[idx], src1[idx]);
                break;
            case 17:
                if (idx < 64) {
                    float x = src0[idx];
                    s32n = idx%31;
                    if (idx == 1) {
                        x = 1;
                        s32n = 129;
                    } else if (idx == 2) {
                        x = 2;
                        s32n = -135;
                    }
                    dst[idx] = ScaLbn(x, s32n);
                    n = (int64_t)s32n;
                    dst[idx+64] = ScaLbn(src0[idx+64], n);
                }
                break;
            case 18:
                dst[idx] = Fma(src0[idx], src1[idx], src2[idx]);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelFloatCompute<T>::Process(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2, const int mode)
{
    AscendC::Simt::VF_CALL<KernelFloatComputeCompute<T>>(Dim3(THREAD_DIM, 1, 1), dst, src0, src1, src2, mode);
}

struct FloatComputeParams {
    int32_t mode;
};

class FloatComputeTestsuite : public testing::Test, public testing::WithParamInterface<FloatComputeParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(FloatComputeTestCase, FloatComputeTestsuite,
    ::testing::Values(FloatComputeParams {6},
    FloatComputeParams {7},
    FloatComputeParams {8},
    FloatComputeParams {9},
    FloatComputeParams {10},
    FloatComputeParams {11},
    FloatComputeParams {12},
    FloatComputeParams {13},
    FloatComputeParams {14},
    FloatComputeParams {15},
    FloatComputeParams {16},
    FloatComputeParams {17},
    FloatComputeParams {18}
                      ));

TEST_P(FloatComputeTestsuite, FloatComputeTestCase)
{
    auto param = GetParam();
    int32_t mode = param.mode;
    int fp_byte_size = 4;
    int shape_size = 128;

    uint8_t dstGm[shape_size * fp_byte_size] = {0};
    uint8_t src0Gm[shape_size * fp_byte_size] = {0};
    uint8_t src1Gm[shape_size * fp_byte_size] = {0};
    uint8_t src2Gm[shape_size * fp_byte_size] = {0};
    KernelFloatCompute<float> op;
    op.Process((__gm__ float*)dstGm, (__gm__ float*)src0Gm, (__gm__ float*)src1Gm, (__gm__ float*)src2Gm, mode);
}

TEST_F(FloatComputeTestsuite, FloatComputeTestCaseAbs)
{
    half x = 123.0;
    half y = -123.0;
    EXPECT_EQ((half)123.0, Simt::Abs(x));
    EXPECT_EQ((half)123.0, Simt::Abs(y));
}

TEST_F(FloatComputeTestsuite, FloatComputeTestCaseFma)
{
    half x = 1.0;
    half y = 2.0;
    half z = 3.0;
    EXPECT_EQ((half)5.0, Simt::Fma(x, y, z));
}
