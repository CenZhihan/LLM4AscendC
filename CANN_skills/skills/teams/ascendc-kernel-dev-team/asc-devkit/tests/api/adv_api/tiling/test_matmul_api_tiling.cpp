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
#define private public
#define protected public
#include "tiling_api.h"
#include "platform_stub.h"
#include "impl/adv_api/tiling/matmul/math_util.h"
#include "impl/adv_api/tiling/matmul/matmul_tiling_algorithm.h"
#include "tiling/platform/platform_ascendc.h"
using namespace AscendC;
using namespace std;
class TestMatmulAPITiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(TestMatmulAPITiling, TestMatmulApiTilingMultiCoreNZINOUT)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(10);
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT, true);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(736, 7168, 4096);
    tilingApi.SetShape(736, 7168, 4096);
    tilingApi.EnableBias(true);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);

    int32_t shapeM = 1;
    int32_t shapeN = 1;
    int32_t shapeK = 1;
    (void)tilingApi.GetSingleShape(shapeM, shapeN, shapeK);
    bool isMAlign = shapeM % 16 == 0 ? true : false;
    bool isNAlign = shapeN % 16 == 0 ? true : false;
    EXPECT_EQ(isMAlign, true);
    EXPECT_EQ(isNAlign, true);
}

TEST_F(TestMatmulAPITiling, TestMatmulApiTilingMultiCoreBTSCM)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(65536, 64, 64);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, TestMatmulApiTilingMultiCoreBTSCM2)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(65536, 64, 64);
    tilingApi.SetShape(65536, 64, 64);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, TestMatmulApiTilingMultiCoreCase)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(4096, 5120, 4096);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, TestMatmulApiTilingMultiCoreCaseWithTranspose)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(4095, 5121, 4095);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, L1CacheUBCase05BothCache)
{
    matmul_tiling::MultiCoreMatmulTiling tiling;
    tiling.SetDim(8);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(-1, -1, -1);
    tiling.SetOrgShape(1024, 5120, 640);
    tiling.EnableBias(true);
    tiling.socVersion = platform_ascendc::SocVersion::ASCEND310P;
    tiling.SetMatmulConfigParams(1, true);
    optiling::TCubeTiling tilingData;
    int ret = tiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tilingData.get_depthAL1CacheUB(), 1);
    EXPECT_EQ(tilingData.get_depthBL1CacheUB(), 2);
}

TEST_F(TestMatmulAPITiling, SmallShapeAlign1)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(256, 512, 4096);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, SmallShapeAlign2)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(32, 6400, 5120);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, SmallShapeAlign3)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(6400, 32, 5120);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, SmallShapeAlign4)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(256, 512, 4096);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);

    tilingApi.SetDim(20);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    tilingApi.SetOrgShape(16, 256, 256);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

    tilingApi.SetOrgShape(16, 256, 256);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

    tilingApi.SetOrgShape(16, 256, 256);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, SmallShapeNotAlign)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(18, 3456, 5120);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}

TEST_F(TestMatmulAPITiling, SmallShapeNotAlign2)
{
    optiling::TCubeTiling tilingData;
    matmul_tiling::MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(24);

    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);

    tilingApi.SetOrgShape(1280, 320, 4);
    tilingApi.SetShape(-1, -1, -1);
    tilingApi.EnableBias(false);

    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    EXPECT_EQ(res, 0);
}



