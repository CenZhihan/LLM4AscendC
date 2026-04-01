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

class TestConv3dFilterTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

void SetType(Conv3dBpFilterTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, ConvCommonApi::ConvDtype::FLOAT32);
    testTiling.SetInputType(TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16);
    testTiling.SetGradOutputType(TPosition::CO1, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16);
}

TEST_F(TestConv3dFilterTiling, Check_Init_001)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    Conv3dBpFilterTiling conv3dDw_empty_init;

    constexpr uint64_t TEST_L1_SIZE_LOCAL = 524288;
    constexpr uint64_t TEST_L0A_SIZE = 65536;
    constexpr uint64_t TEST_L0B_SIZE = 65536;
    constexpr uint64_t TEST_L0C_SIZE = 131072;
    constexpr uint64_t TEST_UB_SIZE = 262144;
    constexpr uint64_t TEST_BT_SIZE = 1024;
    constexpr uint64_t TEST_FB_SIZE = 2048;

    EXPECT_EQ(TEST_L1_SIZE_LOCAL ,conv3dDw_empty_init.platformInfo.l1Size);
    EXPECT_EQ(TEST_L0A_SIZE ,conv3dDw_empty_init.platformInfo.l0ASize);
    EXPECT_EQ(TEST_L0B_SIZE ,conv3dDw_empty_init.platformInfo.l0BSize);
    EXPECT_EQ(TEST_L0C_SIZE ,conv3dDw_empty_init.platformInfo.l0CSize);
    EXPECT_EQ(TEST_UB_SIZE,conv3dDw_empty_init.platformInfo.ubSize);
    EXPECT_EQ(TEST_BT_SIZE,conv3dDw_empty_init.platformInfo.btSize);
    EXPECT_EQ(TEST_FB_SIZE,conv3dDw_empty_init.platformInfo.fbSize);
}

TEST_F(TestConv3dFilterTiling, Check_Init_002)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;

    PlatformInfo platform;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();
    platform.socVersion = socVersion;
    platform.l1Size = static_cast<uint64_t>(0);
    platform.l0ASize = static_cast<uint64_t>(0);
    platform.l0BSize = static_cast<uint64_t>(0);
    platform.l0CSize = static_cast<uint64_t>(0);
    platform.ubSize = static_cast<uint64_t>(0);
    platform.btSize = 0;
    platform.fbSize = 0;
    Conv3dBpFilterTiling conv3dDw_PlatformInfo_init(platform);
    EXPECT_EQ(0 ,conv3dDw_PlatformInfo_init.platformInfo.l1Size);
    EXPECT_EQ(0 ,conv3dDw_PlatformInfo_init.platformInfo.l0ASize);
    EXPECT_EQ(0 ,conv3dDw_PlatformInfo_init.platformInfo.l0BSize);
    EXPECT_EQ(0 ,conv3dDw_PlatformInfo_init.platformInfo.l0CSize);
    EXPECT_EQ(0,conv3dDw_PlatformInfo_init.platformInfo.ubSize);
    EXPECT_EQ(1024,conv3dDw_PlatformInfo_init.platformInfo.btSize);
    EXPECT_EQ(2048,conv3dDw_PlatformInfo_init.platformInfo.fbSize);
}

TEST_F(TestConv3dFilterTiling, Check_Input_Attr)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    Conv3dBpFilterTiling conv3dDw_empty_params;

    // group not 1
    conv3dDw_empty_params.attrInfo.groups = 0;
    EXPECT_EQ(false, conv3dDw_empty_params.CheckInputAttr());
    conv3dDw_empty_params.CheckInputParam();
    conv3dDw_empty_params.attrInfo.groups = 1;

    // pad negative
    conv3dDw_empty_params.attrInfo.padLeft = -1;
    EXPECT_EQ(false, conv3dDw_empty_params.CheckInputAttr());
    conv3dDw_empty_params.attrInfo.padLeft = 0;

    // stride not positive
    conv3dDw_empty_params.attrInfo.strideH = 0;
    EXPECT_EQ(false, conv3dDw_empty_params.CheckInputAttr());
    conv3dDw_empty_params.attrInfo.strideH = 1;

    // dilation not positive
    conv3dDw_empty_params.attrInfo.dilationH = 0;
    EXPECT_EQ(false, conv3dDw_empty_params.CheckInputAttr());
    conv3dDw_empty_params.attrInfo.dilationH = 1;
}

TEST_F(TestConv3dFilterTiling, Check_Input_Shape)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    Conv3dBpFilterTiling conv3dDw_Input_Shape;

    // orgkX
    EXPECT_EQ(false, conv3dDw_Input_Shape.CheckInputShape());
    conv3dDw_Input_Shape.SetWeightShape(240, 17, 1, 3, 3);

    // orgCX
    EXPECT_EQ(false, conv3dDw_Input_Shape.CheckInputShape());
    conv3dDw_Input_Shape.SetInputShape(1, 128, 17, 257, 257);

    // orgOx
    EXPECT_EQ(false, conv3dDw_Input_Shape.CheckInputShape());
    conv3dDw_Input_Shape.SetGradOutputShape(1, 240, 10, 122, 122);
}

TEST_F(TestConv3dFilterTiling, Check_Input_Format)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    Conv3dBpFilterTiling Input_Input_Format;

    Input_Input_Format.descInfo.weightType.format = ConvCommonApi::ConvFormat::NDC1HWC0;
    EXPECT_EQ(false, Input_Input_Format.CheckInputFormat());

    Input_Input_Format.descInfo.weightType.format = ConvCommonApi::ConvFormat::FRACTAL_Z_3D;
    Input_Input_Format.descInfo.fMapType.format = ConvCommonApi::ConvFormat::FRACTAL_Z_3D;
    EXPECT_EQ(false, Input_Input_Format.CheckInputFormat());
}

TEST_F(TestConv3dFilterTiling, Check_LOAD_LIMIT)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);
    conv3dDw.shapeInfo.orgkH = 512;
    EXPECT_EQ(false, conv3dDw.CheckLoad3DLimits());
}

TEST_F(TestConv3dFilterTiling, Check_SetHF32)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);
    conv3dDw.SetHF32(true);
    EXPECT_EQ(true, conv3dDw.hf32Enable_);
    EXPECT_EQ(true, conv3dDw.attrInfo.hf32Enable);
}

TEST_F(TestConv3dFilterTiling, Check_SetGroup)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3dBpFilterTiling conv3dDw(*ascendcPlatform);
    conv3dDw.SetGroup(2);
    EXPECT_EQ(2, conv3dDw.attrInfo.groups);
}

TEST_F(TestConv3dFilterTiling, Check_ShrinkBlockBaseMN)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    ConvBackpropApi::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910B, .l1Size = 524288,
        .l0CSize = 131072, .ubSize = 196608, .l0ASize = 65536, .l0BSize = 65536};
    Conv3dBpFilterTiling conv3dDw(plat);
    conv3dDw.shapeInfo.orgkH = 1;
    conv3dDw.shapeInfo.orgkW = 1;
    conv3dDw.shapeInfo.orgWo = 1;
    conv3dDw.shapeInfo.orgWi = 1;
    conv3dDw.shapeCalc.channelSize = 16;
    conv3dDw.blockTiling_.blockBaseM = 17;
    conv3dDw.blockTiling_.blockBaseN = 16; 
    conv3dDw.blockTiling_.blockBaseK = 17; 
    conv3dDw.blockTiling_.stepKa = 1000; 
    conv3dDw.blockTiling_.stepKb = 1000; 
    conv3dDw.ShrinkBlockBaseMN();
    EXPECT_EQ(16, conv3dDw.blockTiling_.blockBaseM);
    EXPECT_EQ(16, conv3dDw.blockTiling_.blockBaseN);
}


TEST_F(TestConv3dFilterTiling, Check_ShrinkBlockBaseMN_ShouldFurtherAdjustBlockBaseM)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    ConvBackpropApi::PlatformInfo plat {.socVersion = platform_ascendc::SocVersion::ASCEND910B, .l1Size = 524288,
        .l0CSize = 131072, .ubSize = 196608, .l0ASize = 65536, .l0BSize = 65536};
    Conv3dBpFilterTiling conv3dDw(plat);

    conv3dDw.blockTiling_.blockBaseM = 16384;
    conv3dDw.blockTiling_.blockBaseN = 16385; 
    conv3dDw.blockTiling_.blockBaseK = 2;
    
    conv3dDw.shapeInfo.orgkH = 3;
    conv3dDw.shapeInfo.orgkW = 3;
    conv3dDw.shapeInfo.orgWo = 4;
 
    conv3dDw.blockTiling_.stepKb = 4; 
    conv3dDw.ShrinkBlockBaseMN();
    EXPECT_EQ(1008, conv3dDw.blockTiling_.blockBaseM);
    EXPECT_EQ(1008, conv3dDw.blockTiling_.blockBaseN);

    conv3dDw.mmInfo_.kValue = 128;
    conv3dDw.blockTiling_.blockBaseM = 0;
    conv3dDw.ShrinkBaseBlock();
}

TEST_F(TestConv3dFilterTiling, Check_Zero_Filter)
{
    Conv3dBpFilterTiling conv3dDw;
    uint32_t temp = 0;
    EXPECT_EQ(0, conv3dDw.CalculateBl1Cin1CopyLen(temp));
    conv3dDw.shapeInfo.orgkW = 33;
    EXPECT_EQ(false, conv3dDw.CheckInputShape());
    
    conv3dDw.blockTiling_.blockBaseM = 0;
    conv3dDw.mmInfo_.kValue = 128;
    conv3dDw.InitBaseMNK();

    conv3dDw.shapeInfo.orgkW = 0;
    conv3dDw.UpdateSingleCoreInfo();
    conv3dDw.CalculateL1SizeGap();

    conv3dDw.blockTiling_.blockBaseK = 1;
    conv3dDw.blockTiling_.stepN = 0;
    conv3dDw.CalculateL1SizeGap();
}
