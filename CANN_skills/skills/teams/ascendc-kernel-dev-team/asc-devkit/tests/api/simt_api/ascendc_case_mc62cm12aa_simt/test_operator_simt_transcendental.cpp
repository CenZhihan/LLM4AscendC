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
#define DType float

template <typename T>
class KernelTranscendental {
    public:
        __aicore__ KernelTranscendental() {}
        __aicore__ inline void Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__  void KernelTranscendentalCompute(__gm__ T* dst, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    src0[0] = NAN;
    src1[0] = NAN;
    src0[1] = INFINITY;
    src1[1] = INFINITY;
    src0[2] = -INFINITY;
    src1[2] = INFINITY;
    src0[3] = INFINITY;
    src1[3] = -INFINITY;
    src0[4] = INFINITY;
    src0[5] = -INFINITY;
    src1[6] = INFINITY;
    src1[7] = -INFINITY;
    for(int idx = GetThreadIdx<0>() + block_idx*GetThreadNum<0>(); idx < 128; idx+=block_num*GetThreadNum<0>())
    {
        if (mode == 0) {
            dst[idx] = Tan(src0[idx]);
        } else if (mode == 1) {
            dst[idx] = Tanh(src0[idx]);
        } else if (mode == 2) {
            dst[idx] = Tanpi(src0[idx]);
        } else if (mode == 3) {
            dst[idx] = Atan(src0[idx]);
        } else if (mode == 4) {
            dst[idx] = Atan2(src0[idx], src1[idx]);
        } else if (mode == 5) {
            dst[idx] = Atanh(src0[idx]);
        } else if (mode == 6) {
            dst[idx] = Exp(src0[idx]);
        } else if (mode == 7) {
            dst[idx] = Exp2(src0[idx]);
        } else if (mode == 8) {
            dst[idx] = Exp10(src0[idx]);
        } else if (mode == 9) {
            dst[idx] = Expm1(src0[idx]);
        } else if (mode == 10) {
            dst[idx] = Log(src0[idx]);
        } else if (mode == 11) {
            dst[idx] = Log2(src0[idx]);
        } else if (mode == 12) {
            dst[idx] = Log10(src0[idx]);
        } else if (mode == 13) {
            dst[idx] = Log1p(src0[idx]);
        } else if (mode == 14) {
            dst[idx] = Logb(src0[idx]);
        } else if (mode == 15) {
            dst[idx] = Ilogb(src0[idx]);
        } else if (mode == 16) {
            dst[idx] = Pow(src0[idx], src1[idx]);
        } else if (mode == 17) {
            dst[idx] = Cos(src0[idx]);
        } else if (mode == 18) {
            dst[idx] = Cosh(src0[idx]);
        } else if (mode == 19) {
            dst[idx] = Cospi(src0[idx]);
        } else if (mode == 20) {
            dst[idx] = Acos(src0[idx]);
        } else if (mode == 21) {
            dst[idx] = Acosh(src0[idx]);
        } else if (mode == 22) {
            dst[idx] = Sin(src0[idx]);
        } else if (mode == 23) {
            dst[idx] = Sinh(src0[idx]);
        } else if (mode == 24) {
            dst[idx] = Sinpi(src0[idx]);
        } else if (mode == 25) {
            dst[idx] = Asin(src0[idx]);
        } else if (mode == 26) {
            dst[idx] = Asinh(src0[idx]);
        } else if (mode == 27) {
            float s = 0;
            float c = 0;
            Sincos(src0[idx], s, c);
            dst[idx] = s;
        } else if (mode == 28) {
            float s = 0;
            float c = 0;
            Sincos(src0[idx], s, c);
            dst[idx] = c;
        } else if (mode == 29) {
            float s = 0;
            float c = 0;
            Sincospi(src0[idx], s, c);
            dst[idx] = c;
        } else if (mode == 30) {
            float s = 0;
            float c = 0;
            Sincospi(src0[idx], s, c);
            dst[idx] = c;
        }else if (mode == 31) {
            dst[idx] = Sqrt(src0[idx]);
        } else if (mode == 32) {
            dst[idx] = Rsqrt(src0[idx]);
        } else if (mode == 33) {
            int32_t exp=0;
            dst[idx] = Frexp(src0[idx], exp);
        } else if (mode == 34) {
            int exp = src1[idx];
            dst[idx] = Ldexp(src0[idx], exp);
        } else if (mode == 35) {
            dst[idx] = Hypot(src0[idx], src1[idx]);
        } else if (mode == 36) {
            dst[idx] = Rhypot(src0[idx], src1[idx]);
        }else if (mode == 37) {
            int exp = 0;
            Frexp(src0[idx], exp);
            dst[idx] = exp;
        }
    }
}

template <typename T>
__aicore__  inline void KernelTranscendental<T>::Process(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, const int mode)
{
    AscendC::Simt::VF_CALL<KernelTranscendentalCompute<T>>(Dim3(THREAD_DIM, 1, 1), out, src0, src1, mode);
}

struct TranscendentalParams {
    int32_t mode;
};

class TranscendentalTestsuite : public testing::Test, public testing::WithParamInterface<TranscendentalParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TranscendentalTestCase, TranscendentalTestsuite,
    ::testing::Values(TranscendentalParams {0},
    TranscendentalParams {1},
    TranscendentalParams {2},
    TranscendentalParams {3},
    TranscendentalParams {4},
    TranscendentalParams {5},
    TranscendentalParams {6},
    TranscendentalParams {7},
    TranscendentalParams {8},
    TranscendentalParams {9},
    TranscendentalParams {10},
    TranscendentalParams {11},
    TranscendentalParams {12},
    TranscendentalParams {13},
    TranscendentalParams {14},
    TranscendentalParams {15},
    TranscendentalParams {16},
    TranscendentalParams {17},
    TranscendentalParams {18},
    TranscendentalParams {19},
    TranscendentalParams {20},
    TranscendentalParams {21},
    TranscendentalParams {22},
    TranscendentalParams {23},
    TranscendentalParams {24},
    TranscendentalParams {25},
    TranscendentalParams {26},
    TranscendentalParams {27},
    TranscendentalParams {28},
    TranscendentalParams {29},
    TranscendentalParams {30},
    TranscendentalParams {31},
    TranscendentalParams {32},
    TranscendentalParams {33},
    TranscendentalParams {34},
    TranscendentalParams {35},
    TranscendentalParams {36},
    TranscendentalParams {37}
                      ));

TEST_P(TranscendentalTestsuite, TranscendentalTestCase)
{
    auto param = GetParam();
    int32_t mode = param.mode;
    int fp_byte_size = 4;
    int shape_size = 128;

    uint8_t dstGm[shape_size * fp_byte_size] = {0};
    uint8_t src0Gm[shape_size * fp_byte_size] = {0};
    uint8_t src1Gm[shape_size * fp_byte_size] = {0};
    KernelTranscendental<float> op;
    op.Process((__gm__ float*)dstGm, (__gm__ float*)src0Gm, (__gm__ float*)src1Gm, mode);
}

void VerifyFloatNumber(float x, float xExpected, float epsilon = 1e-5)
{
    if (std::isnan(xExpected)) {
        EXPECT_TRUE(std::isnan(x));
    } else if (std::isinf(xExpected)) {
        EXPECT_TRUE(std::isinf(x));
        if (xExpected > 0.0) {
            EXPECT_GT(x, 0.0);
        } else {
            EXPECT_LT(x, 0.0);
        }
    } else {
        EXPECT_NEAR(x, xExpected, epsilon);
    }
}

// ================================ Test Erf start ================================
struct ErfTestParam {
    float x;
    float yExpected;
};
 
class ErfTestSuite : public ::testing::TestWithParam<ErfTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(ErfTestCaseFloat, ErfTestSuite, ::testing::Values(
    ErfTestParam {0.0f, 0.0f},
    ErfTestParam {INFINITY, 1.0f},
    ErfTestParam {-INFINITY, -1.0f},
    ErfTestParam {NAN, NAN},
    ErfTestParam {0.5f, 0.52049988f},
    ErfTestParam {1.5f, 0.96610515f}
));

TEST_P(ErfTestSuite, ErfTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Erf(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Erf end ==================================

// ================================ Test Erfc start ===============================
struct ErfcTestParam {
    float x;
    float yExpected;
};
 
class ErfcTestSuite : public ::testing::TestWithParam<ErfcTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(ErfcTestCaseFloat, ErfcTestSuite, ::testing::Values(
    ErfcTestParam {INFINITY, 0.0f},
    ErfcTestParam {-INFINITY, 2.0f},
    ErfcTestParam {NAN, NAN},
    ErfcTestParam {0.0f, 1.0f},
    ErfcTestParam {0.5f, 0.47950012f},
    ErfcTestParam {10.5f, 7.03592809e-50f}
));

TEST_P(ErfcTestSuite, ErfcTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Erfc(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Erfc end =================================

// ================================ Test Erfinv start =============================
struct ErfinvTestParam {
    float x;
    float yExpected;
};

class ErfinvTestSuite : public ::testing::TestWithParam<ErfinvTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(ErfinvTestCaseFloat, ErfinvTestSuite, ::testing::Values(
    ErfinvTestParam {0.0f, 0.0f},
    ErfinvTestParam {1.0f, INFINITY},
    ErfinvTestParam {-1.0f, -INFINITY},
    ErfinvTestParam {NAN, NAN},
    ErfinvTestParam {2.0f, NAN},
    ErfinvTestParam {-2.0f, NAN},
    ErfinvTestParam {0.5f, 0.47693628f},
    ErfinvTestParam {0.999f, 2.3267564f}
));

TEST_P(ErfinvTestSuite, ErfinvTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Erfinv(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Erfinv end ===============================

// ================================ Test Erfcinv start ============================
struct ErfcinvTestParam {
    float x;
    float yExpected;
};

class ErfcinvTestSuite : public ::testing::TestWithParam<ErfcinvTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(ErfcinvTestCaseFloat, ErfcinvTestSuite, ::testing::Values(
    ErfcinvTestParam {2.0f, -INFINITY},
    ErfcinvTestParam {0.0f, INFINITY},
    ErfcinvTestParam {NAN, NAN},
    ErfcinvTestParam {1.0f, 0.0f},
    ErfcinvTestParam {0.5f, 0.47693628f}
));

TEST_P(ErfcinvTestSuite, ErfcinvTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Erfcinv(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Erfcinv end ==============================

// ================================ Test Erfcx start ==============================
struct ErfcxTestParam {
    float x;
    float yExpected;
};

class ErfcxTestSuite : public ::testing::TestWithParam<ErfcxTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(ErfcxTestCaseFloat, ErfcxTestSuite, ::testing::Values(
    ErfcxTestParam {INFINITY, 0.0f},
    ErfcxTestParam {-INFINITY, INFINITY},
    ErfcxTestParam {NAN, NAN},
    ErfcxTestParam {0.5f, 0.61569034f},
    ErfcxTestParam {-0.5f, 1.95236049f},
    ErfcxTestParam {10.5f, 0.0534919f}
));

TEST_P(ErfcxTestSuite, ErfcxTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Erfcx(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Erfcx end ================================

// ================================ Test Normcdf start ===============================
struct NormcdfTestParam {
    float x;
    float yExpected;
};

class NormcdfTestSuite : public ::testing::TestWithParam<NormcdfTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};

INSTANTIATE_TEST_CASE_P(NormcdfTestCaseFloat, NormcdfTestSuite, ::testing::Values(
    NormcdfTestParam {NAN, NAN},
    NormcdfTestParam {-INFINITY, 0.0f},
    NormcdfTestParam {-14.5f, 0.0f},
    NormcdfTestParam {-2.0f, 0.0227501f},
    NormcdfTestParam {-1.5f, 0.0668072f},
    NormcdfTestParam {-1.0f, 0.158655f},
    NormcdfTestParam {-0.5f, 0.308538f},
    NormcdfTestParam {-0.1f, 0.460172f},
    NormcdfTestParam {-0.052734375f, 0.478972f},
    NormcdfTestParam {-8.248211e-39f, 0.5f},
    NormcdfTestParam {0.0f, 0.5f},
    NormcdfTestParam {8.248211e-39f, 0.5f},
    NormcdfTestParam {0.052734375f, 0.521028f},
    NormcdfTestParam {0.1f, 0.539828f},
    NormcdfTestParam {0.5f, 0.691462f},
    NormcdfTestParam {1.0f, 0.841345f},
    NormcdfTestParam {1.5f, 0.933193f},
    NormcdfTestParam {2.0f, 0.97725f},
    NormcdfTestParam {2.5f, 0.99379f},
    NormcdfTestParam {14.5f, 1.0f},
    NormcdfTestParam {INFINITY, 1.0f}
));

TEST_P(NormcdfTestSuite, NormcdfTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Normcdf(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Normcdf end =================================

// ================================ Test Tgamma start ================================
struct TgammaTestParam {
    float x;
    float yExpected;
};
 
class TgammaTestSuite : public ::testing::TestWithParam<TgammaTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};
 
INSTANTIATE_TEST_CASE_P(TgammaTestCaseFloat, TgammaTestSuite, ::testing::Values(
    TgammaTestParam {0.5f, 1.77245378f},
    TgammaTestParam {-0.5f, -3.54490781f},
    TgammaTestParam {1.0f, 1.0f},
    TgammaTestParam {-1.0f, NAN},
    TgammaTestParam {0.9f, 1.06862879f},
    TgammaTestParam {-0.9f, -10.5705624f},
    TgammaTestParam {9.929099e-39f, 1.00714069e+38f},
    TgammaTestParam {-9.929099e-39f, -1.00714069e+38f},
    TgammaTestParam {3.4e38f, INFINITY},
    TgammaTestParam {3.14f, 2.28448f},
    TgammaTestParam {33.14f, 4.28516609e+35f},
    TgammaTestParam {-33.14f, 5.19573624e-37f},
    TgammaTestParam {0.0f, INFINITY},
    TgammaTestParam {-0.0f, -INFINITY},
    TgammaTestParam {NAN, NAN},
    TgammaTestParam {INFINITY, INFINITY},
    TgammaTestParam {-INFINITY, NAN}
));
 
TEST_P(TgammaTestSuite, TgammaTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Tgamma(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Tgamma end ==================================
 
 
// ================================ Test Lgamma start ================================
struct LgammaTestParam {
    float x;
    float yExpected;
};
 
class LgammaTestSuite : public ::testing::TestWithParam<LgammaTestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};
 
INSTANTIATE_TEST_CASE_P(LgammaTestCaseFloat, LgammaTestSuite, ::testing::Values(
    LgammaTestParam {0.5f, 0.572365f},
    LgammaTestParam {-0.5f, 1.26551199f},
    LgammaTestParam {0.9f, 0.0663762614f},
    LgammaTestParam {-0.9f, 2.35807276f},
    LgammaTestParam {1.0f, 0.0f},
    LgammaTestParam {-1.0f, INFINITY},
    LgammaTestParam {2.14f, 0.0653329268f},
    LgammaTestParam {-2.14f, 1.17242455f},
    LgammaTestParam {3.14f, 0.826139f},
    LgammaTestParam {-3.14f, 0.0282016397f},
    LgammaTestParam {4.5f, 2.45373654f},
    LgammaTestParam {-4.5f, -2.81308413f},
    LgammaTestParam {10.5f, 13.9406261f},
    LgammaTestParam {-10.5f, -15.1472721f},
    LgammaTestParam {3.4e35f, 2.7476849e+37f},
    LgammaTestParam {-3.4e35f, INFINITY},
    LgammaTestParam {-1.25e-20f, 45.8285599f},
    LgammaTestParam {0.0f, INFINITY},
    LgammaTestParam {NAN, NAN},
    LgammaTestParam {INFINITY, INFINITY},
    LgammaTestParam {-INFINITY, INFINITY}
));
 
TEST_P(LgammaTestSuite, LgammaTestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = Lgamma(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test Lgamma end ==================================
 
 
// ================================ Test CylBesselI0 start ================================
struct CylBesselI0TestParam {
    float x;
    float yExpected;
};
 
class CylBesselI0TestSuite : public ::testing::TestWithParam<CylBesselI0TestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};
 
INSTANTIATE_TEST_CASE_P(CylBesselI0TestCaseFloat, CylBesselI0TestSuite, ::testing::Values(
    CylBesselI0TestParam {0.5f, 1.06348336f},
    CylBesselI0TestParam {-0.5f, 1.06348336f},
    CylBesselI0TestParam {3.4e35f, INFINITY},
    CylBesselI0TestParam {13.14f, 56564.625f},
    CylBesselI0TestParam {0.0f, 1.0f},
    CylBesselI0TestParam {NAN, NAN},
    CylBesselI0TestParam {INFINITY, INFINITY},
    CylBesselI0TestParam {-INFINITY, INFINITY}
));
 
TEST_P(CylBesselI0TestSuite, CylBesselI0TestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = CylBesselI0(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test CylBesselI0 end ==================================

 
// ================================ Test CylBesselI1 start ================================
struct CylBesselI1TestParam {
    float x;
    float yExpected;
};
 
class CylBesselI1TestSuite : public ::testing::TestWithParam<CylBesselI1TestParam> {
public:
    void SetUp() override {}
    void TearDown() override {}
};
 
INSTANTIATE_TEST_CASE_P(CylBesselI1TestCaseFloat, CylBesselI1TestSuite, ::testing::Values(
    CylBesselI1TestParam {0.5f, 0.257894307f},
    CylBesselI1TestParam {-0.5f, -0.257894307f},
    CylBesselI1TestParam {3.4e35f, INFINITY},
    CylBesselI1TestParam {13.14f, 54367.7266f},
    CylBesselI1TestParam {0.0f, 0.0f},
    CylBesselI1TestParam {NAN, NAN},
    CylBesselI1TestParam {INFINITY, INFINITY},
    CylBesselI1TestParam {-INFINITY, -INFINITY}
));
 
TEST_P(CylBesselI1TestSuite, CylBesselI1TestCaseFloat)
{
    const auto param = this->GetParam();
    float x = param.x;
    float yExpected = param.yExpected;
    float y = CylBesselI1(x);
    VerifyFloatNumber(y, yExpected);
}
// ================================ Test CylBesselI1 end ==================================
