/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <vector>
#include <gtest/gtest.h>
#include "graph/tensor.h"
#define private public
#define protected public
#define ENABLE_TILING_DEBUG
#include "tiling_api.h"
#include "include/adv_api/conv_backprop/conv3d_bp_filter_tilingdata.h"
#include "include/adv_api/conv_backprop/conv3d_bp_filter_tiling_base.h"
#include "include/adv_api/conv_backprop/conv3d_bp_filter_tiling.h"
#include "include/adv_api/conv/common/conv_common.h"

using namespace std;
using namespace ConvBackpropApi;
using namespace ConvCommonApi;

class TestConv3dFilterTilingNetWork : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

void SetTypeNetWork(Conv3dBpFilterTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT32);
    testTiling.SetInputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
    testTiling.SetGradOutputType(TPosition::CO1, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
}

bool TestBasicBlockResult(uint32_t singleCoreM, uint64_t singleCoreN,
                          uint64_t singleCoreK, optiling::Conv3DBackpropFilterTilingData &tilingData)
{
    bool res = true;
    res = res && singleCoreM == tilingData.basicBlockTiling.get_singleCoreM();
    res = res && singleCoreN == tilingData.basicBlockTiling.get_singleCoreN();
    res = res && singleCoreK == tilingData.basicBlockTiling.get_singleCoreK();
    return res;
}

bool TestBasicBlockResult(uint32_t singleCoreM, uint64_t singleCoreN,
                          uint64_t singleCoreK, AscendC::tiling::Conv3DBackpropFilterTilingData &tilingData)
{
    bool res = true;
    res = res && singleCoreM == tilingData.basicBlockTiling.singleCoreM;
    res = res && singleCoreN == tilingData.basicBlockTiling.singleCoreN;
    res = res && singleCoreK == tilingData.basicBlockTiling.singleCoreK;
    return res;
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_001)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(128, 17, 1, 3, 3);
    conv3dDw.SetInputShape(1, 128, 17, 257, 257);
    conv3dDw.SetGradOutputShape(1, 128, 17, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(128, 1152, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_002)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(3, 64,3, 3, 3);
    conv3dDw.SetInputShape(16, 64, 22, 130, 130);
    conv3dDw.SetGradOutputShape(16, 3, 20, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(16, 576, 512, tilingData), true);

    AscendC::tiling::Conv3DBackpropFilterTilingData tilingData1;
    int64_t ret = conv3dDw.GetTiling(tilingData1);
    EXPECT_EQ(ret, 0);
    conv3dDw.SetWeightShape(128, 17, 1, 3, 0);
    ret = conv3dDw.GetTiling(tilingData1);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_003)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(240, 3, 4, 4, 4);
    conv3dDw.SetInputShape(1, 3, 16, 240, 240);
    conv3dDw.SetGradOutputShape(1, 240, 10, 122, 122);

    conv3dDw.SetPadding(3, 3, 3, 3, 3, 3);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(240, 256, 14896, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_004)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 3, 4, 4, 4);
    conv3dDw.SetInputShape(1, 3, 240, 256, 256);
    conv3dDw.SetGradOutputShape(1, 256, 122, 130, 130);

    conv3dDw.SetPadding(3, 3, 3, 3, 3, 3);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 256, 16912, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_005)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetInputShape(1, 14, 6, 404, 1027);
    conv3dDw.SetGradOutputShape(1, 50, 3, 202, 514);
    conv3dDw.SetWeightShape(50, 14, 20, 1, 1);

    conv3dDw.SetPadding(9, 9, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(64, 16, 103840, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_006)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(64, 186, 3, 1, 1);
    conv3dDw.SetInputShape(2, 186, 3, 727, 1071);
    conv3dDw.SetGradOutputShape(2, 64, 1, 727, 1071);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(64, 192, 1071, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_007)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(1152, 4, 1, 2, 2);
    conv3dDw.SetInputShape(1, 4, 120, 32, 32);
    conv3dDw.SetGradOutputShape(1, 1152, 120, 16, 16);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 64, 256, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_008)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(1152,4, 1, 2, 2);
    conv3dDw.SetInputShape(1, 4, 120, 32, 32);
    conv3dDw.SetGradOutputShape(1, 1152, 120, 16, 16);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 64, 256, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_009)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(128, 128, 1, 3, 3);
    conv3dDw.SetInputShape(1, 128, 17, 257, 257);
    conv3dDw.SetGradOutputShape(2, 128, 17, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(128, 1152, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_010)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(128, 128, 1, 3, 3);
    conv3dDw.SetInputShape(1, 128, 17, 257, 257);
    conv3dDw.SetGradOutputShape(1, 128, 17, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(128, 1152, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_011)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(16, 64, 3, 3, 3);
    conv3dDw.SetInputShape(1, 64, 22, 130, 130);
    conv3dDw.SetGradOutputShape(16, 3, 20, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(16, 576, 512, tilingData), true);

    AscendC::tiling::Conv3DBackpropFilterTilingData tilingData1;
    int64_t ret = conv3dDw.GetTiling(tilingData1);
    EXPECT_EQ(ret, 0);
    conv3dDw.SetWeightShape(128, 17, 1, 3, 0);
    ret = conv3dDw.GetTiling(tilingData1);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_012)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(240, 3, 4, 4, 4);
    conv3dDw.SetInputShape(1, 3, 16, 240, 240);
    conv3dDw.SetGradOutputShape(1, 240, 10, 122, 122);

    conv3dDw.SetPadding(3, 3, 3, 3, 3, 3);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(240, 256, 14896, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_013)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 3, 4, 4, 4);
    conv3dDw.SetInputShape(1, 3, 240, 256, 256);
    conv3dDw.SetGradOutputShape(1, 256, 122, 130, 130);

    conv3dDw.SetPadding(3, 3, 3, 3, 3, 3);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 256, 16912, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_014)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(128, 128, 1, 3, 3);
    conv3dDw.SetInputShape(2, 128, 17, 257, 257);
    conv3dDw.SetGradOutputShape(2, 128, 17, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(128, 1152, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_015)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 128, 1, 1, 1);
    conv3dDw.SetInputShape(1, 128, 1, 128, 128);
    conv3dDw.SetGradOutputShape(1, 256, 1, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 128, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_016)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(2, 48, 1, 1, 1);
    conv3dDw.SetInputShape(8, 48, 96, 96, 96);
    conv3dDw.SetGradOutputShape(8, 2, 96, 96, 96);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(16, 48, 1152, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_017)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 128, 1, 1, 1);
    conv3dDw.SetInputShape(16, 128, 20, 64, 64);
    conv3dDw.SetGradOutputShape(16, 256, 20, 64, 64);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 128, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_018)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(1152, 4, 1, 2, 2);
    conv3dDw.SetInputShape(1, 4, 96, 20, 28);
    conv3dDw.SetGradOutputShape(1, 1152, 96, 10, 14);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 64, 144, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_019)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 128, 1, 1, 1);
    conv3dDw.SetInputShape(1, 128, 1, 128, 128);
    conv3dDw.SetGradOutputShape(1, 256, 1, 128, 128);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 128, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_020)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(2, 48, 1, 1, 1);
    conv3dDw.SetInputShape(8, 48, 96, 96, 96);
    conv3dDw.SetGradOutputShape(8, 2, 96, 96, 96);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(16, 48, 1152, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_021)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(256, 128, 1, 1, 1);
    conv3dDw.SetInputShape(16, 128, 20, 64, 64);
    conv3dDw.SetGradOutputShape(16, 256, 20, 64, 64);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 1, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 128, 512, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_022)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);

    conv3dDw.SetWeightShape(1152, 4, 1, 2, 2);
    conv3dDw.SetInputShape(1, 4, 96, 20, 28);
    conv3dDw.SetGradOutputShape(1, 1152, 96, 10, 14);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(1, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(256, 64, 144, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_023)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    ConvBackpropApi::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910B, .l1Size = 524288,
        .l0CSize = 131072, .ubSize = 196608, .l0ASize = 65536, .l0BSize = 65536};
    Conv3dBpFilterTiling conv3dDw(plat);

    conv3dDw.SetWeightShape(36, 58, 18, 1, 1);
    conv3dDw.SetInputShape(1, 58, 55, 67, 3180);
    conv3dDw.SetGradOutputShape(1, 36, 19, 34, 1590);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 2);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(48, 16, 54064, tilingData), true);
}

TEST_F(TestConv3dFilterTilingNetWork, NetWorks_024)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    ConvBackpropApi::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910B, .l1Size = 524288,
        .l0CSize = 131072, .ubSize = 196608, .l0ASize = 65536, .l0BSize = 65536};
    Conv3dBpFilterTiling conv3dDw(plat);

    conv3dDw.SetWeightShape(64, 56, 2, 1, 1);
    conv3dDw.SetInputShape(21, 56, 150, 2, 753);
    conv3dDw.SetGradOutputShape(21, 64, 75, 1, 753);

    conv3dDw.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDw.SetDilation(1, 1, 1);
    conv3dDw.SetStride(2, 2, 1);
    SetTypeNetWork(conv3dDw);
    int64_t ret1 = conv3dDw.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestBasicBlockResult(64, 64, 768, tilingData), true);
}
