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
#include <random>
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_fp16.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128
template <typename T>
class KernelFP16 {
    public:
        __aicore__ KernelFP16() {}
        __aicore__ inline void Process(__gm__ T* dst, int32_t mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelFP16Compute(__gm__ T* dst, int32_t mode)
{
    for(int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx < 1; idx+=block_num*GetThreadNum<0>()) {
        if(mode == 0) {
            half a = 2.3;
            a++;
            dst[idx] = ++a;
        } else if(mode == 1) {
            half a = 2.1;
            a--;
            dst[idx] = --a;
        } else if(mode == 2) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a && b;
        } else if(mode == 3) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a || b;
        } else if(mode == 4) {
            half a = 2.1;
            dst[idx] = Floor(a);
        } else if(mode == 5) {
            half a = 2.1;
            dst[idx] = Rint(a);
        } else if(mode == 6) {
            half a = 2.1;
            dst[idx] = Ceil(a);
        }
    }
}

template <typename T>
__aicore__ inline void KernelFP16<T>::Process(__gm__ T* dst, int32_t mode)
{
    AscendC::Simt::VF_CALL<KernelFP16Compute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, mode);
}

// ================================ Test half start ===============================
struct FP16Params {
    int32_t mode;
};

class FP16Testsuite : public testing::Test, public testing::WithParamInterface<FP16Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(FP16Testsuite, MathApiTest)
{
    float x = static_cast<float>(rand()) / RAND_MAX;
    half x_h = half(x);

    half result = hcos(x_h);
    half expect = half(cosf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);
    
    result = hsin(x_h);
    expect = half(sinf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = htanh(x_h);
    expect = half(tanhf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hexp(x_h);
    expect = half(expf(x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hexp2(x_h);
    expect = half(exp2f(x_h.ToFloat()));
    EXPECT_EQ(expect, result);
    
    result = hexp10(x_h);
    expect = half(powf(10.0, x_h.ToFloat()));
    EXPECT_EQ(expect, result);

    result = hrcp(x_h);
    if (x_h != half(0)) {
        expect = half(1.0f / x_h.ToFloat());
        EXPECT_EQ(expect, result);
    }
}
// ================================ Test half end ===============================

// ================================ Test half2 start ===============================
struct FP162Params {
    int32_t mode;
};

class FP162Testsuite : public testing::Test, public testing::WithParamInterface<FP162Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(FP162Testsuite, MathApiTest_half2)
{
    float x = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    float y = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    half x_h = half(x);
    half y_h = half(y);
    half2 xy_h2 = {x_h, y_h};

    half2 result = h2cos(xy_h2);
    half expect1 = half(cosf(x_h.ToFloat()));
    half expect2 = half(cosf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2sin(xy_h2);
    expect1 = half(sinf(x_h.ToFloat()));
    expect2 = half(sinf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2tanh(xy_h2);
    expect1 = half(tanhf(x_h.ToFloat()));
    expect2 = half(tanhf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp(xy_h2);
    expect1 = half(expf(x_h.ToFloat()));
    expect2 = half(expf(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp2(xy_h2);
    expect1 = half(exp2f(x_h.ToFloat()));
    expect2 = half(exp2f(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2exp10(xy_h2);
    expect1 = half(powf(10.0, x_h.ToFloat()));
    expect2 = half(powf(10.0, y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2log(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log(x_h.ToFloat()));
        expect2 = half(log(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log2(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log2(x_h.ToFloat()));
        expect2 = half(log2(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log10(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(log10(x_h.ToFloat()));
        expect2 = half(log10(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rcp(xy_h2);
    if (x_h != half(0) && y_h != half(0)) {
        expect1 = half(1.0f / x_h.ToFloat());
        expect2 = half(1.0f / y_h.ToFloat());
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }
    
    result = h2sqrt(xy_h2);
    if (x_h > half(0) && y_h > half(0)) {
        expect1 = half(sqrt(x_h.ToFloat()));
        expect2 = half(sqrt(y_h.ToFloat()));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rsqrt(xy_h2);
    if (x_h > half(0) &&y_h > half(0)) {
        expect1 = half(1.0f / sqrt(x_h.ToFloat()));
        expect2 = half(1.0f / sqrt(y_h.ToFloat()));
    }
    
    result = h2ceil(xy_h2);
    expect1 = half(ceil(x_h.ToFloat()));
    expect2 = half(ceil(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2floor(xy_h2);
    expect1 = half(floor(x_h.ToFloat()));
    expect2 = half(floor(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2rint(xy_h2);
    expect1 = half(rint(x_h.ToFloat()));
    expect2 = half(rint(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2trunc(xy_h2);
    expect1 = half(trunc(x_h.ToFloat()));
    expect2 = half(trunc(y_h.ToFloat()));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
}

TEST_F(FP162Testsuite, CastApiTest_half2)
{

    float x = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    float y = static_cast<float>(rand()) / RAND_MAX + 0.1f;
    half x_h = half(x);
    half y_h = half(y);
    half2 xy_h2_1 = {x_h, y_h};
    half2 xy_h2_2 = {x_h, y_h};
    float2 xy_f322 = {x, y};

    half2 result = __floats2half2_rn(x, y);
    half expect1 = half(x);
    half expect2 = half(y);
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = __float22half2_rn(xy_f322);
    expect1 = half(xy_f322.x);
    expect2 = half(xy_f322.y);
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    float res_32 = __low2float(xy_h2_1);
    float expect_fp32 = xy_h2_1.x.ToFloat();
    EXPECT_EQ(expect_fp32, res_32);

    half res_fp16 = __low2half(xy_h2_1);
    EXPECT_EQ(xy_h2_1.x, res_fp16);

    result = __low2half2(xy_h2_1);
    expect1 = x_h;
    expect2 = x_h;
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = __lowhigh2highlow(xy_h2_1);
    expect1 = x_h;
    expect2 = y_h;
    EXPECT_EQ(expect1, result.y);
    EXPECT_EQ(expect2, result.x);

    res_32 = __high2float(xy_h2_1);
    expect_fp32 = y_h.ToFloat();
    EXPECT_EQ(expect_fp32, res_32);

    res_fp16 = __high2half(xy_h2_1);
    EXPECT_EQ(y_h, res_fp16);

    result = __high2half2(xy_h2_1);
    EXPECT_EQ(y_h, result.x);
    EXPECT_EQ(y_h, result.y);

    result = __highs2half2(xy_h2_1, xy_h2_2);
    expect1 = half(xy_h2_1.y);
    expect2 = half(xy_h2_2.y);
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = __lows2half2(xy_h2_1, xy_h2_2);
    expect1 = half(xy_h2_1.x);
    expect2 = half(xy_h2_2.x);
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = __halves2half2(x_h, y_h);
    expect1 = x_h;
    expect2 = y_h;
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    res_fp16  = __ushort_as_half((unsigned short)0x3C00U);
    EXPECT_EQ((half)1.0, res_fp16);
}
// ================================ Test half2 end ===============================

// ================================ Test cast start ===============================
class SimtCastFp16Testsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(SimtCastFp16Testsuite, CastFp16Test)
{
    uint32_t src0 = 4099;
    half dst0 = 0.0;
    dst0 = __uint2half_rn_sat(src0);
    EXPECT_EQ((half)4100.0, dst0);

    dst0 = __uint2half_rz_sat(src0);
    EXPECT_EQ((half)4096.0, dst0);

    dst0 = __uint2half_rd_sat(src0);
    EXPECT_EQ((half)4096.0, dst0);

    dst0 = __uint2half_ru_sat(src0);
    EXPECT_EQ((half)4100.0, dst0);

    dst0 = __uint2half_rna_sat(src0);
    EXPECT_EQ((half)4100.0, dst0);

    src0 = 100000;
    dst0 = __uint2half_rna_sat(src0);
    EXPECT_EQ((half)65504.0, dst0);

    int32_t src1 = 4099;
    half dst1 = 0.0;
    dst1 = __int2half_rn_sat(src1);
    EXPECT_EQ((half)4100.0, dst1);

    dst1 = __int2half_rz_sat(src1);
    EXPECT_EQ((half)4096.0, dst1);

    dst1 = __int2half_rd_sat(src1);
    EXPECT_EQ((half)4096.0, dst1);

    dst1 = __int2half_ru_sat(src1);
    EXPECT_EQ((half)4100.0, dst1);

    dst1 = __int2half_rna_sat(src1);
    EXPECT_EQ((half)4100.0, dst1);

    src1 = 100000;
    dst1 = __int2half_rna_sat(src1);
    EXPECT_EQ((half)65504.0, dst1);

    float src3 = 1.5f;
    float dst3 = 0.0f;
    dst3 = __float2float_rn(src3);
    EXPECT_EQ(2.0f, dst3);

    dst3 = __float2float_rz(src3);
    EXPECT_EQ(1.0f, dst3);

    dst3 = __float2float_rd(src3);
    EXPECT_EQ(1.0f, dst3);

    dst3 = __float2float_ru(src3);
    EXPECT_EQ(2.0f, dst3);

    dst3 = __float2float_rna(src3);
    EXPECT_EQ(2.0f, dst3);

    float src4 = 0.5 + pow(2.0, -12);
    half dst4 = 0.0;
    dst4 = __float2half_rn_sat(src4);
    EXPECT_EQ((half)0.5, dst4);

    dst4 = __float2half_rz_sat(src4);
    EXPECT_EQ((half)0.5, dst4);

    dst4 = __float2half_rd_sat(src4);
    EXPECT_EQ((half)0.5, dst4);

    dst4 = __float2half_ru_sat(src4);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst4);

    dst4 = __float2half_rna_sat(src4);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst4);

    dst4 = __float2half_ro_sat(src4);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst4);

    src4 = 100000.f;
    dst4 = __float2half_rna_sat(src4);
    EXPECT_EQ((half)65504.0, dst4);

    src4 = 0.5 + pow(2.0, -12);
    float2 src5 = {src4, src4};
    half2 dst5 = {0.0, 0.0};

    dst5 = __float22half2_rn(src5);
    EXPECT_EQ((half)0.5, dst5.x);

    dst5 = __float22half2_rd(src5);
    EXPECT_EQ((half)0.5, dst5.x);

    dst5 = __float22half2_ru(src5);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst5.x);

    dst5 = __float22half2_rna(src5);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst5.x);

    float2 src6 = {src4, src4};
    half2 dst6 = {0.0, 0.0};

    dst6 = __float22half2_rn_sat(src6);
    EXPECT_EQ((half)0.5, dst6.x);

    dst6 = __float22half2_rz_sat(src6);
    EXPECT_EQ((half)0.5, dst6.x);

    dst6 = __float22half2_rd_sat(src6);
    EXPECT_EQ((half)0.5, dst6.x);

    dst6 = __float22half2_ru_sat(src6);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst6.x);

    dst6 = __float22half2_rna_sat(src6);
    EXPECT_EQ((half)(0.5 + pow(2.0, -11)), dst6.x);
}
// ================================ Test cast end ===============================