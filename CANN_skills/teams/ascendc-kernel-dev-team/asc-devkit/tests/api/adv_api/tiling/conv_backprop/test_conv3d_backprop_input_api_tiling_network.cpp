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

void SetType(Conv3DBpInputTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT16);
    testTiling.SetGradOutputType(TPosition::CO1, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
    testTiling.SetInputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
}

void SetFp32Type(Conv3DBpInputTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT32);
    testTiling.SetGradOutputType(TPosition::CO1, ConvFormat::NDC1HWC0, ConvDtype::FLOAT32);
    testTiling.SetInputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::FLOAT32);
}

bool TestSingleCoreResult(uint32_t singleCoreBatch, uint32_t singleCoreGroup, uint32_t singleCoreDin, uint32_t singleCoreHo,
                    uint32_t singleCoreM, uint32_t singleCoreCout, uint32_t singleCoreCout1,
                    uint32_t singleCoreCin, uint32_t singleCoreCin1,
                    optiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && singleCoreBatch == tilingData.conv3DDxTiling.get_singleCoreBatch(); 
    res = res && singleCoreGroup == tilingData.conv3DDxTiling.get_singleCoreGroup();
    res = res && singleCoreDin == tilingData.conv3DDxTiling.get_singleCoreDin();
    res = res && singleCoreHo == tilingData.conv3DDxTiling.get_singleCoreHo();
    res = res && singleCoreM == tilingData.conv3DDxTiling.get_singleCoreM();
    res = res && singleCoreCout == tilingData.conv3DDxTiling.get_singleCoreCout();
    res = res && singleCoreCout1 == tilingData.conv3DDxTiling.get_singleCoreCout1();
    res = res && singleCoreCin == tilingData.conv3DDxTiling.get_singleCoreCin();
    res = res && singleCoreCin1 == tilingData.conv3DDxTiling.get_singleCoreCin1();
    return res;
}                    

bool TestSingleCoreResult(uint32_t singleCoreBatch, uint32_t singleCoreGroup, uint32_t singleCoreDin, uint32_t singleCoreHo,
                    uint32_t singleCoreM, uint32_t singleCoreCout, uint32_t singleCoreCout1,
                    uint32_t singleCoreCin, uint32_t singleCoreCin1,
                    AscendC::tiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && singleCoreBatch == tilingData.conv3DDxTiling.singleCoreBatch; 
    res = res && singleCoreGroup == tilingData.conv3DDxTiling.singleCoreGroup;
    res = res && singleCoreDin == tilingData.conv3DDxTiling.singleCoreDin;
    res = res && singleCoreHo == tilingData.conv3DDxTiling.singleCoreHo;
    res = res && singleCoreM == tilingData.conv3DDxTiling.singleCoreM;
    res = res && singleCoreCout == tilingData.conv3DDxTiling.singleCoreCout;
    res = res && singleCoreCout1 == tilingData.conv3DDxTiling.singleCoreCout1;
    res = res && singleCoreCin == tilingData.conv3DDxTiling.singleCoreCin;
    res = res && singleCoreCin1 == tilingData.conv3DDxTiling.singleCoreCin1;
    return res;
} 

bool TestBaseResult(uint32_t baseM, uint32_t baseK, uint32_t baseN,
                    uint32_t baseD, uint32_t baseBatch, uint32_t baseGroup,
                    optiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && baseM == tilingData.conv3DDxTiling.get_baseM();
    res = res && baseK == tilingData.conv3DDxTiling.get_baseK();
    res = res && baseN == tilingData.conv3DDxTiling.get_baseN();
    res = res && baseD == tilingData.conv3DDxTiling.get_baseD();
    res = res && baseBatch == tilingData.conv3DDxTiling.get_baseBatch();
    res = res && baseGroup == tilingData.conv3DDxTiling.get_baseGroup(); 
    return res;
}

bool TestBaseResult(uint32_t baseM, uint32_t baseK, uint32_t baseN,
                    uint32_t baseD, uint32_t baseBatch, uint32_t baseGroup,
                    AscendC::tiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && baseM == tilingData.conv3DDxTiling.baseM;
    res = res && baseK == tilingData.conv3DDxTiling.baseK;
    res = res && baseN == tilingData.conv3DDxTiling.baseN;
    res = res && baseD == tilingData.conv3DDxTiling.baseD;
    res = res && baseBatch == tilingData.conv3DDxTiling.baseBatch;
    res = res && baseGroup == tilingData.conv3DDxTiling.baseGroup; 
    return res;
}

bool TestPbBufferResult(uint32_t al0Pbuffer, uint32_t bl0Pbuffer, uint32_t cl0Pbuffer,
                    uint32_t al1Pbuffer, uint32_t bl1Pbuffer,
                    optiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && al0Pbuffer == tilingData.conv3DDxTiling.get_al0Pbuffer();
    res = res && bl0Pbuffer == tilingData.conv3DDxTiling.get_bl0Pbuffer();
    res = res && cl0Pbuffer == tilingData.conv3DDxTiling.get_cl0Pbuffer();
    res = res && al1Pbuffer == tilingData.conv3DDxTiling.get_al1Pbuffer();
    res = res && bl1Pbuffer == tilingData.conv3DDxTiling.get_bl1Pbuffer();
    return res;
}

bool TestPbBufferResult(uint32_t al0Pbuffer, uint32_t bl0Pbuffer, uint32_t cl0Pbuffer,
                    uint32_t al1Pbuffer, uint32_t bl1Pbuffer,
                    AscendC::tiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && al0Pbuffer == tilingData.conv3DDxTiling.al0Pbuffer;
    res = res && bl0Pbuffer == tilingData.conv3DDxTiling.bl0Pbuffer;
    res = res && cl0Pbuffer == tilingData.conv3DDxTiling.cl0Pbuffer;
    res = res && al1Pbuffer == tilingData.conv3DDxTiling.al1Pbuffer;
    res = res && bl1Pbuffer == tilingData.conv3DDxTiling.bl1Pbuffer;
    return res;
}

bool TestStepResult(uint32_t stepM, uint32_t stepN, uint32_t stepKa,
                    uint32_t stepKb, uint32_t stepBatch, uint32_t stepGroup,
                    optiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && stepM == tilingData.conv3DDxTiling.get_stepM();
    res = res && stepN == tilingData.conv3DDxTiling.get_stepN();
    res = res && stepKa == tilingData.conv3DDxTiling.get_stepKa();
    res = res && stepKb == tilingData.conv3DDxTiling.get_stepKb();
    res = res && stepBatch == tilingData.conv3DDxTiling.get_stepBatch();
    res = res && stepGroup == tilingData.conv3DDxTiling.get_stepGroup(); 
    return res;
}

bool TestStepResult(uint32_t stepM, uint32_t stepN, uint32_t stepKa,
                    uint32_t stepKb, uint32_t stepBatch, uint32_t stepGroup,
                    AscendC::tiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && stepM == tilingData.conv3DDxTiling.stepM;
    res = res && stepN == tilingData.conv3DDxTiling.stepN;
    res = res && stepKa == tilingData.conv3DDxTiling.stepKa;
    res = res && stepKb == tilingData.conv3DDxTiling.stepKb;
    res = res && stepBatch == tilingData.conv3DDxTiling.stepBatch;
    res = res && stepGroup == tilingData.conv3DDxTiling.stepGroup; 
    return res;
}

bool TestOtherResult(uint32_t iterateOrder, uint32_t hf32Flag,
                     optiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && iterateOrder == tilingData.conv3DDxTiling.get_iterateOrder();
    res = res && hf32Flag == tilingData.conv3DDxTiling.get_hf32Flag();
    return res;
}

bool TestOtherResult(uint32_t iterateOrder, uint32_t hf32Flag,
                     AscendC::tiling::Conv3DBackpropInputTilingData &tilingData)
{
    bool res = true;
    res = res && iterateOrder == tilingData.conv3DDxTiling.iterateOrder;
    res = res && hf32Flag == tilingData.conv3DDxTiling.hf32Flag;
    return res;
}

// x1_17
TEST_F(TestConv3dInputTiling, NetWorks_001)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(512, 512, 1, 1, 1);
    conv3dDx.SetInputShape(1, 512, 5, 32, 32);
    conv3dDx.SetGradOutputShape(1, 512, 5, 32, 32);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 1024, 512, 32, 128, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 64, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 4, 8, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);

    AscendC::tiling::Conv3DBackpropInputTilingData tilingData1;
    ret1 = conv3dDx.GetTiling(tilingData1);
    EXPECT_EQ(ret1, 0);
    conv3dDx.SetWeightShape(512, 512, 1, 1, 0);
    ret1 = conv3dDx.GetTiling(tilingData1);
    EXPECT_EQ(ret1, -1);
}

// magvit_1
TEST_F(TestConv3dInputTiling, NetWorks_002)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(3, 64, 3, 3, 3);
    conv3dDx.SetInputShape(16, 64, 22, 130, 130);
    conv3dDx.SetGradOutputShape(16, 3, 20, 128, 128);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16900, 16, 1, 64, 4, tilingData), true);
    EXPECT_EQ(TestBaseResult(336, 48, 64, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 3, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// magvit_2
TEST_F(TestConv3dInputTiling, NetWorks_003)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(64, 64, 1, 1, 1);
    conv3dDx.SetInputShape(16, 64, 20, 128, 128);
    conv3dDx.SetGradOutputShape(16, 64, 20, 128, 128);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 64, 4, 64, 4, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 64, 64, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 1, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_10
TEST_F(TestConv3dInputTiling, NetWorks_004)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(256, 128, 3, 3, 3);
    conv3dDx.SetInputShape(1, 128, 11, 128, 128);
    conv3dDx.SetGradOutputShape(1, 256, 9, 128, 128);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 256, 16, 128, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 48, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 21, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_12
TEST_F(TestConv3dInputTiling, NetWorks_005)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(256, 256, 3, 3, 3);
    conv3dDx.SetInputShape(1, 256, 11, 128, 128);
    conv3dDx.SetGradOutputShape(1, 256, 9, 128, 128);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 256, 16, 128, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 48, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 21, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_14
TEST_F(TestConv3dInputTiling, NetWorks_006)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(3, 128, 3, 3, 3);
    conv3dDx.SetInputShape(1, 128, 19, 256, 256);
    conv3dDx.SetGradOutputShape(1, 3, 17, 256, 256);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 65536, 16, 1, 128, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 48, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 3, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// svd_7
TEST_F(TestConv3dInputTiling, NetWorks_007)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(1280, 1280, 3, 1, 1);
    conv3dDx.SetInputShape(1, 1280, 25, 5, 8);
    conv3dDx.SetGradOutputShape(1, 1280, 25, 5, 8);

    conv3dDx.SetPadding(1, 1, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 48, 1280, 80, 256, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(48, 64, 256, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 18, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_bs2_20
TEST_F(TestConv3dInputTiling, NetWorks_008)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(512, 256, 1, 1, 1);
    conv3dDx.SetInputShape(2, 256, 5, 64, 64);
    conv3dDx.SetGradOutputShape(2, 512, 5, 64, 64);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 4096, 512, 32, 128, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 64, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 4, 8, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// videogpt_h240_3
TEST_F(TestConv3dInputTiling, NetWorks_009)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(240, 4, 1, 1, 1);
    conv3dDx.SetInputShape(1, 4, 4, 60, 60);
    conv3dDx.SetGradOutputShape(1, 240, 4, 60, 60);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 3600, 240, 15, 16, 1, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 64, 16, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 4, 4, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_17_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_001)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(512, 512, 1, 1, 1);
    conv3dDx.SetInputShape(1, 512, 5, 32, 32);
    conv3dDx.SetGradOutputShape(1, 512, 5, 32, 32);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 1024, 512, 64, 128, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 32, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 2, 16, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// magvit_1_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_002)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(3, 64, 3, 3, 3);
    conv3dDx.SetInputShape(16, 64, 22, 130, 130);
    conv3dDx.SetGradOutputShape(16, 3, 20, 128, 128);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16900, 8, 1, 64, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(336, 24, 64, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 3, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// magvit_2_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_003)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(64, 64, 1, 1, 1);
    conv3dDx.SetInputShape(16, 64, 20, 128, 128);
    conv3dDx.SetGradOutputShape(16, 64, 20, 128, 128);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 64, 8, 64, 8, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 32, 64, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 2, 2, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_10_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_004)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(256, 128, 3, 3, 3);
    conv3dDx.SetInputShape(1, 128, 11, 128, 128);
    conv3dDx.SetGradOutputShape(1, 256, 9, 128, 128);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 256, 32, 128, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 24, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 21, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_12_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_005)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(256, 256, 3, 3, 3);
    conv3dDx.SetInputShape(1, 256, 11, 128, 128);
    conv3dDx.SetGradOutputShape(1, 256, 9, 128, 128);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 16384, 256, 32, 128, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 24, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 21, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_14_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_006)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(3, 128, 3, 3, 3);
    conv3dDx.SetInputShape(1, 128, 19, 256, 256);
    conv3dDx.SetGradOutputShape(1, 3, 17, 256, 256);

    conv3dDx.SetPadding(0, 0, 1, 1, 1, 1);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 65536, 8, 1, 128, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 24, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 3, 3, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// svd_7_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_007)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(1280, 1280, 3, 1, 1);
    conv3dDx.SetInputShape(1, 1280, 25, 5, 8);
    conv3dDx.SetGradOutputShape(1, 1280, 25, 5, 8);

    conv3dDx.SetPadding(1, 1, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 40, 1280, 160, 256, 32, tilingData), true);
    EXPECT_EQ(TestBaseResult(40, 32, 256, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 2, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 26, 2, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// x1_bs2_20_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_008)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(512, 256, 1, 1, 1);
    conv3dDx.SetInputShape(2, 256, 5, 64, 64);
    conv3dDx.SetGradOutputShape(2, 512, 5, 64, 64);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 4096, 512, 64, 128, 16, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 32, 128, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 2, 16, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

// videogpt_h240_3_fp32
TEST_F(TestConv3dInputTiling, NetWorks_fp32_009)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(240, 4, 1, 1, 1);
    conv3dDx.SetInputShape(1, 4, 4, 60, 60);
    conv3dDx.SetGradOutputShape(1, 240, 4, 60, 60);

    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetBackpropPadding(0, 0, 0, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 1, 1);
    conv3dDx.SetHF32(false);
    conv3dDx.SetGroup(1);
    SetFp32Type(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
    EXPECT_EQ(TestSingleCoreResult(1, 1, 1, 1, 3600, 240, 30, 8, 1, tilingData), true);
    EXPECT_EQ(TestBaseResult(256, 32, 8, 1, 1, 1, tilingData), true);
    EXPECT_EQ(TestPbBufferResult(2, 2, 1, 2, 1, tilingData), true);
    EXPECT_EQ(TestStepResult(1, 1, 4, 8, 1, 1, tilingData), true);
    EXPECT_EQ(TestOtherResult(1, 0, tilingData), true);
}

TEST_F(TestConv3dInputTiling, NetWorks_bf16_convtranspose_001)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    Conv3DBpInputTiling conv3dDx(*ascendcPlatform);

    conv3dDx.SetWeightShape(44, 4, 2, 3, 2);
    conv3dDx.SetGradOutputShape(1, 44, 17, 5, 77);
    
    conv3dDx.SetPadding(0, 0, 0, 0, 0, 0);
    conv3dDx.SetOutputPadding(0, 1, 0);
    conv3dDx.SetDilation(1, 1, 1);
    conv3dDx.SetStride(1, 2, 1);
    
    SetType(conv3dDx);

    int64_t ret1 = conv3dDx.GetTiling(tilingData);
    EXPECT_EQ(ret1, 0);
}
