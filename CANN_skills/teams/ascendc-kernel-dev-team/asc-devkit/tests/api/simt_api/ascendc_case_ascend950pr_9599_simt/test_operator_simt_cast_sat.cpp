/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

namespace {
enum class ROUND { R, A, F, C, Z, O, H};

enum class RoundingSaturation {
    RS_DISABLE_VALUE = 0,
    RS_ENABLE_VALUE = 1,
};

enum class RoundMode : uint8_t {
    CAST_NONE = 0,
    CAST_RINT, // round
    CAST_FLOOR,
    CAST_CEIL,
    CAST_ROUND,  // away-zero
    CAST_TRUNC,  // to-zero
    CAST_ODD,    // Von Neumann rounding
    CAST_HYBRID, // hybrid round
    CAST_EVEN,
    CAST_ZERO,
    UNKNOWN = 0xFF,
};

template <typename... Arg>
struct Tuple {};

typedef uint16_t half;
typedef uint16_t bfloat16_t;
}

template <ROUND rnd, RoundingSaturation sat, typename T>
int32_t __cvt_int32_t(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
uint32_t __cvt_uint32_t(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
int64_t __cvt_int64_t(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
uint64_t __cvt_uint64_t(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
half __cvt_half(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
float __cvt_float(T x)
{
    return 0;
}
template <ROUND rnd, RoundingSaturation sat, typename T>
bfloat16_t __cvt_bfloat16_t(T x)
{
    return 0;
}

#define __SIMT_DEVICE_FUNCTIONS_DECL__ 

#undef ASCENDC_CPU_DEBUG
#include "impl/simt_api/cpp/dav_3510/kernel_simt_cast_sat_impl.h"

class CastSatTestsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(CastSatTestsuite, CastTestCase)
{
    float x = 1.0f;
    int32_t ret = AscendC::Simt::CastSat<int32_t, float, RoundMode::CAST_RINT>(x);
    EXPECT_EQ(0, ret);
}