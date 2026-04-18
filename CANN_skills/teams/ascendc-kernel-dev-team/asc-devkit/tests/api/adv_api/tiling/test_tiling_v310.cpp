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
#include "graph/tensor.h"
#include <dlfcn.h>
#define private public
#define protected public
#include "include/adv_api/activation/softmax_tiling.h"
#include "tiling_api.h"
#include "platform_stub.h"
#include "impl/adv_api/tiling/matmul/math_util.h"
#include "impl/adv_api/tiling/matmul/matmul_tiling_algorithm.h"
#include "tiling/platform/platform_ascendc.h"
using namespace AscendC;
using namespace ge;
using namespace std;
using namespace matmul_tiling;

class TestTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(TestTiling, TestMxMatmulFP4NDTiling)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E1M2);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E1M2);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(64, 1088, 64);
    tiling.SetOrgShape(64, 1088, 64);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP4NZTiling)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT4_E2M1, true);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT4_E2M1, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(128, 4096, 7168);
    tiling.SetOrgShape(128, 4096, 7168);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP8NDTilingCase1)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(64, 1088, 64);
    tiling.SetOrgShape(64, 1088, 64);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP8NDTilingCase2)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(8960, 5120, 2048);
    tiling.SetOrgShape(8960, 5120, 2048);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, MultiCoreSmallMNFP4)
{
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3,rnnMatmul4,rnnMatmul5;
    rnnMatmul3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT4_E1M2);
    rnnMatmul3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1,-1,-1,-1,-1,-1);
    rnnMatmul3.EnableMultiCoreSplitK(true);
    auto ret = rnnMatmul3.EnableBias(true);
    ret = rnnMatmul3.SetDim(32);
    ret = rnnMatmul3.SetOrgShape(5, 40, 986);
    ret = rnnMatmul3.SetShape(5, 10, 986);
    ret = rnnMatmul3.SetBufferSpace(-1, -1, -1); // will use all buffer space if not explicitly specified
    optiling::TCubeTiling tilingData;
    ret = rnnMatmul3.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, MultiCoreSmallMNFP8)
{
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul3,rnnMatmul4,rnnMatmul5;
    rnnMatmul3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT8_E4M3FN);
    rnnMatmul3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType ::DT_FLOAT);
    rnnMatmul3.SetSingleRange(-1,-1,-1,-1,-1,-1);
    rnnMatmul3.EnableMultiCoreSplitK(true);
    auto ret = rnnMatmul3.EnableBias(true);
    ret = rnnMatmul3.SetDim(32);
    ret = rnnMatmul3.SetOrgShape(5, 40, 986);
    ret = rnnMatmul3.SetShape(5, 10, 986);
    ret = rnnMatmul3.SetBufferSpace(-1, -1, -1); // will use all buffer space if not explicitly specified
    optiling::TCubeTiling tilingData;
    ret = rnnMatmul3.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulFP8NDTiling_CaseTscm)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::TSCM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::TSCM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(1, 1920, 64);
    tiling.SetOrgShape(1, 1920, 64);
    
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulTilingCeilKIsOdd)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, true);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 392;
    int32_t n = 320;
    int32_t k = 672;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);

    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tilingData.get_stepM(), 1);
    EXPECT_EQ(tilingData.get_depthA1(), 1);
}

TEST_F(TestTiling, TestMxMatmulTilingEnableL1BankConflictOptimise)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 1024;
    int32_t n = 1024;
    int32_t k = 1024;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulTilingScaleBNoTransEnableL1BankConflictOptimise)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E4M3FN, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::NZ, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 1024;
    int32_t n = 1024;
    int32_t k = 1024;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 1024;
    int32_t n = 1024;
    int32_t k = 1024;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetDequantType(DequantType::TENSOR);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16TSCM)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::TSCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::TSCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 256;
    int32_t n = 256;
    int32_t k = 256;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP16NORM)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 768;
    int32_t n = 768;
    int32_t k = 768;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetMatmulConfigParams(0);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMatmulTilingEnableL1BankConflictOptimiseFP32)
{
    MultiCoreMatmulTiling tiling;
    tiling.SetDim(1);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    int32_t m = 16;
    int32_t n = 16;
    int32_t k = 16;

    tiling.SetShape(m, n, k);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    int ret1 = tiling.EnableL1BankConflictOptimise();

    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestBitwiseNotTiling)
{
    fe::PlatFormInfos platformInfo;
    const platform_ascendc::PlatformAscendC plat = platform_ascendc::PlatformAscendC(&platformInfo);
    uint32_t maxVal = 0;
    uint32_t minVal = 0;
    std::vector<int64_t> shapeDims = { 128 };
    auto shape = ge::Shape(shapeDims);
    GetBitwiseNotMaxMinTmpSize(plat, shape, sizeof(float), false, maxVal, minVal);
    EXPECT_EQ(maxVal, 0);
    EXPECT_EQ(minVal, 0);
    uint32_t extraBuf = 0;
    uint32_t maxLivedNodesCnt = 0;
    GetBitwiseNotTmpBufferFactorSize(plat, sizeof(float), maxLivedNodesCnt, extraBuf);
    EXPECT_EQ(extraBuf, 0);
    EXPECT_EQ(maxLivedNodesCnt, 0);
}

TEST_F(TestTiling, TestBitwiseOrTiling)
{
    fe::PlatFormInfos platformInfo;
    const platform_ascendc::PlatformAscendC plat = platform_ascendc::PlatformAscendC(&platformInfo);
    uint32_t maxVal = 0;
    uint32_t minVal = 0;
    std::vector<int64_t> shapeDims = { 128 };
    auto shape = ge::Shape(shapeDims);
    GetBitwiseOrMaxMinTmpSize(plat, shape, sizeof(float), false, maxVal, minVal);
    EXPECT_EQ(maxVal, 0);
    EXPECT_EQ(minVal, 0);
    uint32_t extraBuf = 0;
    uint32_t maxLivedNodesCnt = 0;
    GetBitwiseOrTmpBufferFactorSize(plat, sizeof(float), maxLivedNodesCnt, extraBuf);
    EXPECT_EQ(extraBuf, 0);
    EXPECT_EQ(maxLivedNodesCnt, 0);
}

TEST_F(TestTiling, TestBitwiseXorTiling)
{
    fe::PlatFormInfos platformInfo;
    const platform_ascendc::PlatformAscendC plat = platform_ascendc::PlatformAscendC(&platformInfo);
    uint32_t maxVal = 0;
    uint32_t minVal = 0;
    std::vector<int64_t> shapeDims = { 128 };
    auto shape = ge::Shape(shapeDims);
    GetBitwiseXorMaxMinTmpSize(plat, shape, sizeof(float), false, maxVal, minVal);
    EXPECT_EQ(maxVal, 0);
    EXPECT_EQ(minVal, 0);
    uint32_t extraBuf = 0;
    uint32_t maxLivedNodesCnt = 0;
    GetBitwiseXorTmpBufferFactorSize(plat, sizeof(float), maxLivedNodesCnt, extraBuf);
    EXPECT_EQ(extraBuf, 0);
    EXPECT_EQ(maxLivedNodesCnt, 0);
}

TEST_F(TestTiling, TestLogicalXorTiling)
{
    fe::PlatFormInfos platformInfo;
    uint32_t typeSize = 2u;
    auto plat = platform_ascendc::PlatformAscendC(&platformInfo);
    auto LogicalXorShape = ge::Shape({128,128});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    GetLogicalXorMaxMinTmpSize(plat, LogicalXorShape, sizeof(float), false, maxValue, minValue);
    EXPECT_EQ(minValue, 0);
    EXPECT_EQ(maxValue, 0);

    uint32_t maxLivedNodesCnt = 0;
    uint32_t extraBuf = 0;
    GetLogicalXorTmpBufferFactorSize(plat, sizeof(float), maxLivedNodesCnt, extraBuf);
    EXPECT_EQ(maxLivedNodesCnt, 0);
    EXPECT_EQ(extraBuf, 0);
    
    GetLogicalXorMaxMinTmpSize(plat, LogicalXorShape, typeSize, false, maxValue, minValue);
    EXPECT_EQ(minValue, 0);
    EXPECT_EQ(maxValue, 0);

    GetLogicalXorTmpBufferFactorSize(plat, typeSize, maxLivedNodesCnt, extraBuf);
    EXPECT_EQ(maxLivedNodesCnt, 0);
    EXPECT_EQ(extraBuf, 0);
}

TEST_F(TestTiling, TestMxMatmulGemvFP4BaseNAlign64Case)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::VECTOR, matmul_tiling::DataType::DT_FLOAT4_E1M2, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT4_E1M2, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::VECTOR, false);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::ND, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(1, 26, 5056);
    tiling.SetOrgShape(1, 26, 5056);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tilingData.get_baseM(), 16);
    EXPECT_EQ(tilingData.get_baseN(), 32);
    EXPECT_EQ(tilingData.get_baseK(), 2048);
    EXPECT_EQ(tilingData.get_dbL0A(), 2);
    // fp4 baseN align to 64, baseN * baseK * dbL0B = 64K
    EXPECT_EQ(tilingData.get_dbL0B(), 1);
}

TEST_F(TestTiling, TestMxMatmulGemvbaseKalign1024Case)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::VECTOR, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::VECTOR, false);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::ND, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(1, 256, 17552);
    tiling.SetOrgShape(1, 256, 17552);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);

    // MX Gemv baseK must align 1024,
    EXPECT_EQ(tilingData.get_baseK(), 1024);
}

TEST_F(TestTiling, TestMxMatmulL1)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::TSCM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetBType(TPosition::TSCM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetScaleAType(TPosition::TSCM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::TSCM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(256, 256, 256);
    tiling.SetOrgShape(256, 256, 256);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulUB)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::VECOUT, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetBType(TPosition::VECOUT, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetScaleAType(TPosition::VECOUT, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::VECOUT, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::VECOUT, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(256, 256, 256);
    tiling.SetOrgShape(256, 256, 256);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulMNInnerNZSmallSize)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(32, 32, 32);
    tiling.SetOrgShape(32, 32, 32);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulL1GMLargeSize)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::TSCM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::TSCM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::GM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(608, 192, 768);
    tiling.SetOrgShape(608, 192, 768);
    tiling.SetBias(false);
    // A/scaleA is TSCM, remained L1Size: 524288 - 608*768 - 608*768/32 = 42752
    tiling.SetBufferSpace(42752, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestTiling, TestMxMatmulGML1LargeSize)
{
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    tiling.SetBType(TPosition::TSCM, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    tiling.SetScaleAType(TPosition::GM, CubeFormat::NZ, false);
    tiling.SetScaleBType(TPosition::TSCM, CubeFormat::NZ, true);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tiling.SetShape(192, 672, 736);
    tiling.SetOrgShape(192, 672, 736);
    tiling.SetBias(false);
    // B/scaleB is TSCM, remained L1Size: 524288 - 736*672 - 936*672/32 = 14240
    tiling.SetBufferSpace(14240, -1, -1, -1);
    tiling.SetFixSplit(-1, -1, -1);
    tiling.SetMadType(MatrixMadType::MXMODE);

    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}