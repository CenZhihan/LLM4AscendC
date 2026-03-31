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
#include "include/adv_api/conv_backprop/conv3d_bp_input_tilingdata.h"
#include "include/adv_api/conv_backprop/conv3d_bp_input_tiling.h"
#include "include/adv_api/conv/common/conv_common.h"

using namespace std;
using namespace ConvBackpropApi;
using namespace ConvCommonApi;

class TestConv3dInputTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
    uint64_t l1Size = 524288;
    uint64_t l0aSize = 65536;
    uint64_t l0bSize = 65536;
    uint64_t l0cSize = 262144;
};

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201))
TEST_F(TestConv3dInputTiling, Init_Check_001)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx_empty_init;

    constexpr uint64_t TEST_L1_SIZE_LOCAL = 524288;
    constexpr uint64_t TEST_L0A_SIZE = 65536;
    constexpr uint64_t TEST_L0B_SIZE = 65536;
    constexpr uint64_t TEST_L0C_SIZE = 131072;
    constexpr uint64_t TEST_UB_SIZE = 262144;
    constexpr uint64_t TEST_BT_SIZE = 1024;
    constexpr uint64_t TEST_FB_SIZE = 2048;

    EXPECT_EQ(TEST_L1_SIZE_LOCAL, conv3dDx_empty_init.platformInfo.l1Size);
    EXPECT_EQ(TEST_L0A_SIZE, conv3dDx_empty_init.platformInfo.l0ASize);
    EXPECT_EQ(TEST_L0B_SIZE, conv3dDx_empty_init.platformInfo.l0BSize);
    EXPECT_EQ(TEST_L0C_SIZE, conv3dDx_empty_init.platformInfo.l0CSize);
    EXPECT_EQ(TEST_UB_SIZE, conv3dDx_empty_init.platformInfo.ubSize);
    EXPECT_EQ(TEST_BT_SIZE, conv3dDx_empty_init.platformInfo.btSize);
    EXPECT_EQ(TEST_FB_SIZE, conv3dDx_empty_init.platformInfo.fbSize);
}

TEST_F(TestConv3dInputTiling, Init_Check_002)
{
    optiling::Conv3DBackpropInputTilingData tilingData;

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
    Conv3DBpInputTiling  conv3dDx_PlatformInfo_init(platform);

    EXPECT_EQ(0, conv3dDx_PlatformInfo_init.platformInfo.l1Size);
    EXPECT_EQ(0, conv3dDx_PlatformInfo_init.platformInfo.l0ASize);
    EXPECT_EQ(0, conv3dDx_PlatformInfo_init.platformInfo.l0BSize);
    EXPECT_EQ(0, conv3dDx_PlatformInfo_init.platformInfo.l0CSize);
    EXPECT_EQ(0, conv3dDx_PlatformInfo_init.platformInfo.ubSize);
    EXPECT_EQ(1024, conv3dDx_PlatformInfo_init.platformInfo.btSize);
    EXPECT_EQ(2048, conv3dDx_PlatformInfo_init.platformInfo.fbSize);
}

TEST_F(TestConv3dInputTiling, Input_Attr_Check)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx_empty_params;

    // group not 1
    conv3dDx_empty_params.attrInfo.groups = 0;
    EXPECT_EQ(false, conv3dDx_empty_params.CheckInputAttr());
    conv3dDx_empty_params.CheckInputParam();
    conv3dDx_empty_params.attrInfo.groups = 1;

    // pad negative
    conv3dDx_empty_params.attrInfo.padLeft = -1;
    EXPECT_EQ(false, conv3dDx_empty_params.CheckInputAttr());
    conv3dDx_empty_params.attrInfo.padLeft = 1;

    // backpropPad negative
    conv3dDx_empty_params.attrInfo.backpropPadLeft = -1;
    EXPECT_EQ(false, conv3dDx_empty_params.CheckInputAttr());
    conv3dDx_empty_params.attrInfo.backpropPadLeft = 1;

    // stride not positive
    conv3dDx_empty_params.attrInfo.strideH = 0;
    EXPECT_EQ(false, conv3dDx_empty_params.CheckInputAttr());
    conv3dDx_empty_params.attrInfo.strideH = 1;

    // dilation not positive
    conv3dDx_empty_params.attrInfo.dilationH = 0;
    EXPECT_EQ(false, conv3dDx_empty_params.CheckInputAttr());
    conv3dDx_empty_params.attrInfo.dilationH = 1;
}

TEST_F(TestConv3dInputTiling, Input_Shape_Check)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx_Input_Shape;

    EXPECT_EQ(false, conv3dDx_Input_Shape.CheckInputShape());
    conv3dDx_Input_Shape.SetWeightShape(512, 240, 1, 3, 3);
    EXPECT_EQ(false, conv3dDx_Input_Shape.CheckInputShape());
    conv3dDx_Input_Shape.SetGradOutputShape(1, 240, 10, 122, 122);
    EXPECT_EQ(false, conv3dDx_Input_Shape.CheckInputShape());
    conv3dDx_Input_Shape.SetInputShape(1, 512, 5, 32, 32);
}

TEST_F(TestConv3dInputTiling, Invalid_attrs)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(512, 512, 1, 1, 1);
    conv3dDx.SetInputShape(1, 512, 5, 32, 32);
    conv3dDx.SetGradOutputShape(1, 512, 5, 32, 32);
    conv3dDx.SetDilation(1, 1, 1);

    conv3dDx.SetStride(256, 256, 256);
    EXPECT_EQ(false, conv3dDx.CheckAttrs());

    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetPadding(256, 256, 0, 0, 0, 0);
    EXPECT_EQ(false, conv3dDx.CheckPadRange());
}

TEST_F(TestConv3dInputTiling, Check_Input_Format)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx_Input_Format;

    conv3dDx_Input_Format.descInfo.weightType.format = ConvCommonApi::ConvFormat::NDC1HWC0;
    EXPECT_EQ(false, conv3dDx_Input_Format.CheckInputFormat());

    conv3dDx_Input_Format.descInfo.outBackpropType.format = ConvCommonApi::ConvFormat::FRACTAL_Z_3D;
    conv3dDx_Input_Format.descInfo.fMapType.format = ConvCommonApi::ConvFormat::FRACTAL_Z_3D;
    EXPECT_EQ(false, conv3dDx_Input_Format.CheckInputFormat());
}

TEST_F(TestConv3dInputTiling, ShrinkBasicBlock)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx;

    conv3dDx.tilingParams.baseM = 1024;
    conv3dDx.tilingParams.baseN = 326;
    conv3dDx.tilingParams.baseK = 18;

    conv3dDx.tilingParams.stepKa = 1;
    conv3dDx.tilingParams.stepKb = 2;

    conv3dDx.shapeInfo.orgkW = 1;
    conv3dDx.shapeInfo.orgkH = 1;

    conv3dDx.shapeInfo.orgWi = 32;
    conv3dDx.tilingParams.al0Pbuffer = 2;
    conv3dDx.tilingParams.al1Pbuffer = 2;
    conv3dDx.tilingParams.bl1Pbuffer = 2;
    conv3dDx.mmInfo_.kValue = 32;
    conv3dDx.lenHkWkC0_ = 1;

    conv3dDx.ShrinkBasicBlock();
    EXPECT_EQ(1024, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(326, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(18, conv3dDx.tilingParams.baseK);
}

TEST_F(TestConv3dInputTiling, LegalProtection)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    Conv3DBpInputTiling conv3dDx;

    conv3dDx.tilingParams.stepKa = 7;
    conv3dDx.tilingParams.stepKb = 3;

    conv3dDx.tilingParams.baseM = 3;
    conv3dDx.tilingParams.baseN = 5;
    conv3dDx.tilingParams.baseK = 1;

    conv3dDx.shapeInfo.orgkW = 1;
    conv3dDx.shapeInfo.orgkH = 1;

    conv3dDx.shapeInfo.orgWi = 32;
    conv3dDx.tilingParams.al0Pbuffer = 2;
    conv3dDx.tilingParams.al1Pbuffer = 2;
    conv3dDx.tilingParams.bl1Pbuffer = 2;
    conv3dDx.mmInfo_.kValue = 32;
    

    conv3dDx.lenHkWkC0_ = 2;
    conv3dDx.LegalProtection();
    EXPECT_EQ(3, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(5, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(16, conv3dDx.tilingParams.baseK);

    conv3dDx.lenHkWkC0_ = 3;
    conv3dDx.LegalProtection();
    EXPECT_EQ(3, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(5, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(16, conv3dDx.tilingParams.baseK);

    conv3dDx.tilingParams.al1Pbuffer = 1;
    conv3dDx.tilingParams.bl1Pbuffer = 2;
    conv3dDx.LegalProtection();
    EXPECT_EQ(3, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(5, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(16, conv3dDx.tilingParams.baseK);

    conv3dDx.tilingParams.al1Pbuffer = 2;
    conv3dDx.tilingParams.bl1Pbuffer = 1;
    conv3dDx.LegalProtection();
    EXPECT_EQ(3, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(5, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(16, conv3dDx.tilingParams.baseK);
}

TEST_F(TestConv3dInputTiling, ShrinkBasicBlock_ShouldRestoreOriginalValues_WhenAllAdjustmentsFail)
{
    Conv3DBpInputTiling conv3dDx;

    conv3dDx.tilingParams.baseM = 49;
    conv3dDx.tilingParams.baseN = 32;
    conv3dDx.tilingParams.baseK = 100;

    conv3dDx.lenHkWkC0_ = 3;
    conv3dDx.blockSize_ = 64;

    conv3dDx.shapeInfo.orgWi = 50;
    conv3dDx.shapeInfo.orgWo = 524288;
    conv3dDx.shapeInfo.orgHo = 2;

    conv3dDx.tilingParams.al0Pbuffer = 2;
    conv3dDx.tilingParams.bl0Pbuffer = 2;
    conv3dDx.tilingParams.al1Pbuffer = 1;
    conv3dDx.tilingParams.bl1Pbuffer = 1;

    conv3dDx.shapeInfo.orgkW = 4;
    conv3dDx.shapeInfo.orgkH = 4;

    conv3dDx.mmInfo_.kValue = 200;


    conv3dDx.tilingParams.stepKa = 2;
    conv3dDx.tilingParams.stepKb = 2;
    conv3dDx.tilingParams.stepN = 1;
    conv3dDx.tilingParams.stepM = 1;
    
    conv3dDx.ShrinkBasicBlock();

    EXPECT_EQ(49, conv3dDx.tilingParams.baseM);
    EXPECT_EQ(32, conv3dDx.tilingParams.baseN);
    EXPECT_EQ(100, conv3dDx.tilingParams.baseK);

    conv3dDx.lenHkWkC0_ = 0;
    conv3dDx.ShrinkBasicBlock();
}


TEST_F(TestConv3dInputTiling, Check_Zero_Input)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);
    conv3dDx.tilingParams.baseM = 0;
    conv3dDx.tilingParams.baseN = 1;
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
    conv3dDx.shapeInfo.orgWo = 1;
    conv3dDx.lenHkWkC0_ = 1;
    conv3dDx.dtypeByte_ = 2;
    conv3dDx.tilingParams.bl1Pbuffer = 1;
    conv3dDx.tilingParams.al1Pbuffer = 1;
    conv3dDx.attrInfo.strideW = 1;
    conv3dDx.blockSize_ = 16;
    conv3dDx.EqualL1MatchStepMNK(stepKa, stepKb);

    uint32_t baseM = 0;
    uint32_t baseN = 4;
    uint32_t baseK = 4;
    conv3dDx.mmInfo_.mValue = 1;
    conv3dDx.mmInfo_.kValue = 1;
    conv3dDx.mmInfo_.nValue = 1;
    conv3dDx.AdjustBaseMNK(1, 1, baseM, baseN, baseK);
    
    
    conv3dDx.shapeInfo.orgkH = 1;
    conv3dDx.shapeInfo.orgkW = 1;
    conv3dDx.tilingParams.baseK = 1;
    uint32_t stepKa_temp = 1;
    EXPECT_EQ(false, conv3dDx.IsStepL1Valid(stepKa_temp, stepKb));

    conv3dDx.lenHkWkC0_ = 0;
    EXPECT_EQ(false, conv3dDx.IsStepL1Valid(stepKa_temp, stepKb));
    conv3dDx.AdjustBaseMNK(1, 1, baseM, baseN, baseK);
    conv3dDx.EqualL1MatchStepMNK(stepKa, stepKb);
}
#endif
