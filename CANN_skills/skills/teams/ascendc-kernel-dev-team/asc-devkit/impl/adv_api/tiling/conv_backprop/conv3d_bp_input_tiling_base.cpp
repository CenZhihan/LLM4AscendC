/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/*!
 * \file conv3d_bp_input_tiling_base.cpp
 * \brief
 */
#include "../../detail/host_log.h"
#include "conv3d_bp_tiling_util.h"
#include "include/adv_api/conv_backprop/conv3d_bp_input_tiling_base.h"

namespace ConvBackpropApi {

constexpr uint64_t L0A_SIZE = 65536;
constexpr uint64_t L0B_SIZE = 65536;
constexpr uint64_t L0C_SIZE = 131072;
constexpr uint64_t UB_SIZE = 262144;
constexpr uint64_t BT_SIZE = 1024;
constexpr uint64_t FB_SIZE = 2048;
constexpr uint64_t L1_SIZE_LOCAL = 524288;

Conv3DBpInputTilingBase::Conv3DBpInputTilingBase()
{
    this->platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    this->platformInfo.l1Size = L1_SIZE_LOCAL;
    this->platformInfo.l0ASize = L0A_SIZE;
    this->platformInfo.l0BSize = L0B_SIZE;
    this->platformInfo.l0CSize = L0C_SIZE;
    this->platformInfo.ubSize = UB_SIZE;
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

Conv3DBpInputTilingBase::Conv3DBpInputTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
{
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize);

    this->platformInfo.socVersion = socVersion;
    this->platformInfo.l1Size = static_cast<uint64_t>(l1Size);
    this->platformInfo.l0ASize = static_cast<uint64_t>(l0ASize);
    this->platformInfo.l0BSize = static_cast<uint64_t>(l0BSize);
    this->platformInfo.l0CSize = static_cast<uint64_t>(l0CSize);
    this->platformInfo.ubSize = static_cast<uint64_t>(ubSize);
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

Conv3DBpInputTilingBase::Conv3DBpInputTilingBase(const PlatformInfo& platform)
{
    this->platformInfo.socVersion = platform.socVersion;
    this->platformInfo.l1Size = static_cast<uint64_t>(platform.l1Size);
    this->platformInfo.l0ASize = static_cast<uint64_t>(platform.l0ASize);
    this->platformInfo.l0BSize = static_cast<uint64_t>(platform.l0BSize);
    this->platformInfo.l0CSize = static_cast<uint64_t>(platform.l0CSize);
    this->platformInfo.ubSize = static_cast<uint64_t>(platform.ubSize);
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

bool Conv3DBpInputTilingBase::SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
{
    if (this->shapeInfo.orgCi > 0 && this->shapeInfo.orgCi != cin) {
        TILING_LOG_ERROR(
            "inChannels(cin) has been set a different value: curValue=%ld, oldValue=%ld", cin, this->shapeInfo.orgCi);
        return false;
    }
    if (this->shapeInfo.orgCo > 0 && this->shapeInfo.orgCo != cout) {
        TILING_LOG_ERROR(
            "outChannels(cout) has been set a different value: curValue=%ld, oldValue=%ld", cout, this->shapeInfo.orgCo);
        return false;
    }

    this->shapeInfo.orgkH = h;
    this->shapeInfo.orgkW = w;
    this->shapeInfo.orgkD = d;
    this->shapeInfo.orgCi = cin;
    this->shapeInfo.orgCo = cout;
    return true;
}

bool Conv3DBpInputTilingBase::SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if (this->shapeInfo.orgCi > 0 && this->shapeInfo.orgCi != c) {
        TILING_LOG_ERROR(
            "channels(c) has been set a different value: curValue=%ld, oldValue=%ld", c, this->shapeInfo.orgCi);
        return false;
    }
    if (this->shapeInfo.orgN > 0 && this->shapeInfo.orgN != n) {
        TILING_LOG_ERROR(
            "batch(n) has been set a different value: curValue=%ld, oldValue=%ld", n, this->shapeInfo.orgN);
        return false;
    }

    this->shapeInfo.orgN = n;
    this->shapeInfo.orgCi = c;
    this->shapeInfo.orgHi = h;
    this->shapeInfo.orgWi = w;
    this->shapeInfo.orgDi = d;
    this->opType_ = OpType::kConv3DBackpropInput; // transpose不会设置此接口
    return true;
}

bool Conv3DBpInputTilingBase::SetGradOutputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    if (this->shapeInfo.orgN > 0 && this->shapeInfo.orgN != n) {
         TILING_LOG_ERROR(
            "batch(n) has been set a different value: curValue=%ld, oldValue=%ld", n, this->shapeInfo.orgN);
        return false;
    }
    if (this->shapeInfo.orgCo > 0 && this->shapeInfo.orgCo != c) {
        TILING_LOG_ERROR(
            "channels(c) has been set a different value: curValue=%ld, oldValue=%ld", c, this->shapeInfo.orgCo);
        return false;
    }

    this->shapeInfo.orgN = n;
    this->shapeInfo.orgCo = c;
    this->shapeInfo.orgDo = d;
    this->shapeInfo.orgHo = h;
    this->shapeInfo.orgWo = w;
    return true;
}

void Conv3DBpInputTilingBase::SetWeightType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.weightType.pos = pos;
    this->descInfo.weightType.format = format;
    this->descInfo.weightType.dtype = dtype;
}

void Conv3DBpInputTilingBase::SetInputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.fMapType.pos = pos;
    this->descInfo.fMapType.format = format;
    this->descInfo.fMapType.dtype = dtype;
}

void Conv3DBpInputTilingBase::SetGradOutputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.outBackpropType.pos = pos;
    this->descInfo.outBackpropType.format = format;
    this->descInfo.outBackpropType.dtype = dtype;
}


void Conv3DBpInputTilingBase::SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight)
{
    this->attrInfo.padFront = padFront;
    this->attrInfo.padBack = padBack;
    this->attrInfo.padUp = padUp;
    this->attrInfo.padDown = padDown;
    this->attrInfo.padLeft = padLeft;
    this->attrInfo.padRight = padRight;
}

void Conv3DBpInputTilingBase::SetBackpropPadding(int64_t backpropPadUp, int64_t backpropPadDown,
        int64_t backpropPadLeft, int64_t backpropPadRight)
{
    this->attrInfo.backpropPadUp = backpropPadUp;
    this->attrInfo.backpropPadDown = backpropPadDown;
    this->attrInfo.backpropPadLeft = backpropPadLeft;
    this->attrInfo.backpropPadRight = backpropPadRight;
}

void Conv3DBpInputTilingBase::SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
{
    this->attrInfo.dilationH = dilationH;
    this->attrInfo.dilationW = dilationW;
    this->attrInfo.dilationD = dilationD;
}

void Conv3DBpInputTilingBase::SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
{
    this->attrInfo.strideH = strideH;
    this->attrInfo.strideW = strideW;
    this->attrInfo.strideD = strideD;
}

bool Conv3DBpInputTilingBase::SetOutputPadding(int64_t outputPadD, int64_t outputPadH, int64_t outputPadW)
{
    if (this->opType_ == OpType::kConv3DBackpropInput) {
        TILING_LOG_ERROR(
            "The interface SetOutputPadding conflicts with SetInputShape, To implement Conv3dTranspose, not call SetInputShape please.");  
        return false;
    }
    this->attrInfo.outputPadH = outputPadH;
    this->attrInfo.outputPadW = outputPadW;
    this->attrInfo.outputPadD = outputPadD;
    return true;
}

void Conv3DBpInputTilingBase::SetHF32(uint8_t hf32Enable)
{
    this->attrInfo.hf32Enable = hf32Enable;
}

void Conv3DBpInputTilingBase::SetGroup(int64_t groups)
{
    // this should be called after SetOrgWeightShape and SetOrgFmapShape
    this->attrInfo.groups = groups;
}

bool Conv3DBpInputTilingBase::ShapeInitCalc()
{
    int64_t k0 = C0_BYTE_SIZE / g_dtypeSizeTab.at(this->descInfo.fMapType.dtype);
    int64_t yShapeC0 = k0;
    int64_t fmapShapeC0 = k0;
    shapeCalc.Ci1 = CeilDivision(shapeInfo.orgCi, fmapShapeC0);
    shapeCalc.Co1 = CeilDivision(shapeInfo.orgCo, yShapeC0);

    int64_t magFactor0 = Lcm(shapeInfo.orgCi / attrInfo.groups, k0) / (shapeInfo.orgCi / attrInfo.groups);
    int64_t magFactor1 = Lcm(shapeInfo.orgCo / attrInfo.groups, yShapeC0) / (shapeInfo.orgCo / attrInfo.groups);
    int64_t magFactor = Min(Lcm(magFactor0, magFactor1), attrInfo.groups);
    shapeCalc.realG = (attrInfo.groups + magFactor - 1) / magFactor;
    shapeCalc.cin1G = (magFactor * shapeInfo.orgCi / attrInfo.groups + fmapShapeC0 - 1) / fmapShapeC0;
    shapeCalc.cout1G = (magFactor * shapeInfo.orgCo / attrInfo.groups + yShapeC0 - 1) / yShapeC0;
    return true;
}

bool Conv3DBpInputTilingBase::CheckInputParam()
{
    if (!CheckInputAttr() || !CheckInputShape() || !CheckInputFormat()) {
        return false;
    }
    return true;
}

bool Conv3DBpInputTilingBase::CheckInputAttr()
{
    if (this->attrInfo.groups != 1){
        TILING_LOG_ERROR(
            "Illegal attrs have set: groups=%ld.", this->attrInfo.groups);
        return false;
    }

    bool padInvalidFlag = (this->attrInfo.padLeft < 0 || this->attrInfo.padRight < 0 ||
        this->attrInfo.padUp < 0 || this->attrInfo.padDown < 0 ||
        this->attrInfo.padFront < 0 || this->attrInfo.padBack < 0);
    if (padInvalidFlag) {
        TILING_LOG_ERROR(
            "Illegal attrs have set: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld, padFront=%ld, padBack=%ld,\
            which must >= 0.", this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft,
            this->attrInfo.padRight, this->attrInfo.padFront, this->attrInfo.padBack);
        return false;
    }

    bool outBackpropPadInvalidFlag = (this->attrInfo.backpropPadLeft < 0 || this->attrInfo.backpropPadRight < 0 ||
        this->attrInfo.backpropPadUp < 0 || this->attrInfo.backpropPadDown < 0);
    if (outBackpropPadInvalidFlag) {
        TILING_LOG_ERROR(
            "Illegal attrs have set: backpropPadLeft=%ld, backpropPadDown=%ld, backpropPadLeft=%ld, backpropPadRight=%ld,\
            which must >= 0.", this->attrInfo.backpropPadUp, this->attrInfo.backpropPadDown, this->attrInfo.backpropPadLeft,
            this->attrInfo.backpropPadRight);
        return false;
    }

    if (this->attrInfo.strideH <= 0 || this->attrInfo.strideW <= 0 || this->attrInfo.strideD <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: strideH=%ld, strideW=%ld, strideD=%ld, which must > 0.",
                         this->attrInfo.strideH, this->attrInfo.strideW, this->attrInfo.strideD);
        return false;
    }

    if (this->attrInfo.dilationH <= 0 || this->attrInfo.dilationW <= 0 || this->attrInfo.dilationD <= 0) {
        TILING_LOG_ERROR("Illegal attrs have set: dilationH=%ld, dilationW=%ld, dilationD=%ld, which must > 0.",
                         this->attrInfo.dilationH, this->attrInfo.dilationW, this->attrInfo.dilationD);
        return false;
    }

    return true;
}

bool Conv3DBpInputTilingBase::CheckInputShape()
{
    if (this->shapeInfo.orgkH <= 0 || this->shapeInfo.orgkW <= 0 || this->shapeInfo.orgkD <= 0) {
        TILING_LOG_ERROR("Illegal org weight shapes have set: kH=%ld, kW=%ld, kD=%ld, which must > 0.",
            this->shapeInfo.orgkH, this->shapeInfo.orgkW, this->shapeInfo.orgkD);
        return false;
    }

    if (this->shapeInfo.orgN <= 0 || this->shapeInfo.orgCo <= 0 || this->shapeInfo.orgHo <= 0 ||
        this->shapeInfo.orgWo <= 0 || this->shapeInfo.orgDo <= 0) {
        TILING_LOG_ERROR("Illegal org outBackprop shapes have set: N=%ld Co=%ld, Ho=%ld, Wo=%ld, Do=%ld, which must > 0.",
            this->shapeInfo.orgN, this->shapeInfo.orgCo, this->shapeInfo.orgHo, this->shapeInfo.orgWo, this->shapeInfo.orgDo);
        return false;
    }

    return true;
}

bool Conv3DBpInputTilingBase::CheckInputFormat()
{
    if (this->descInfo.weightType.format != ConvCommonApi::ConvFormat::FRACTAL_Z_3D) {
        TILING_LOG_ERROR("Unsupported weight format: %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }

    if (this->descInfo.outBackpropType.format != ConvCommonApi::ConvFormat::NDC1HWC0) {
        TILING_LOG_ERROR("Unsupported outBackprop format: %s.",
                         g_formatToStr.at(this->descInfo.outBackpropType.format).c_str());
        return false;
    }

    return true;
}

void Conv3DBpInputTilingBase::PrintTilingData() const
{
    TILING_LOG_DEBUG("batch: %ld", shapeInfo.orgN);
    TILING_LOG_DEBUG("cin: %ld", shapeInfo.orgCi);
    TILING_LOG_DEBUG("di: %ld", shapeInfo.orgDi);
    TILING_LOG_DEBUG("hi: %ld", shapeInfo.orgHi);
    TILING_LOG_DEBUG("wi: %ld", shapeInfo.orgWi);
    TILING_LOG_DEBUG("dk: %ld", shapeInfo.orgkD);
    TILING_LOG_DEBUG("hk: %ld", shapeInfo.orgkH);
    TILING_LOG_DEBUG("wk: %ld", shapeInfo.orgkW);
    TILING_LOG_DEBUG("cout: %ld", shapeInfo.orgCo);
    TILING_LOG_DEBUG("dout: %ld", shapeInfo.orgDo);
    TILING_LOG_DEBUG("ho: %ld", shapeInfo.orgHo);
    TILING_LOG_DEBUG("wo: %ld", shapeInfo.orgWo);
    TILING_LOG_DEBUG("cin1G: %ld", shapeCalc.cin1G);
    TILING_LOG_DEBUG("cout1G: %ld", shapeCalc.cout1G);
    TILING_LOG_DEBUG("group: %ld", attrInfo.groups);
    TILING_LOG_DEBUG("padFront: %ld", attrInfo.padFront);
    TILING_LOG_DEBUG("padBack:  %ld", attrInfo.padBack);
    TILING_LOG_DEBUG("padUp: %ld", attrInfo.padUp);
    TILING_LOG_DEBUG("padDown: %ld", attrInfo.padDown);
    TILING_LOG_DEBUG("padLeft: %ld", attrInfo.padLeft);
    TILING_LOG_DEBUG("padRight: %ld", attrInfo.padRight);
    TILING_LOG_DEBUG("outBackpropPadUp: %ld", attrInfo.backpropPadUp);
    TILING_LOG_DEBUG("outBackpropPadDown: %ld", attrInfo.backpropPadDown);
    TILING_LOG_DEBUG("outBackpropPadLeft: %ld", attrInfo.backpropPadLeft);
    TILING_LOG_DEBUG("outBackpropPadRight: %ld", attrInfo.backpropPadRight);
    TILING_LOG_DEBUG("strideH: %ld", attrInfo.strideH);
    TILING_LOG_DEBUG("strideW: %ld", attrInfo.strideW);
    TILING_LOG_DEBUG("strideD: %ld", attrInfo.strideD);
    TILING_LOG_DEBUG("dilationH : %ld", attrInfo.dilationH);
    TILING_LOG_DEBUG("dilationW : %ld", attrInfo.dilationW);
    TILING_LOG_DEBUG("dilationD : %ld", attrInfo.dilationD);
    TILING_LOG_DEBUG("hf32Flag: %u", attrInfo.hf32Enable);
}

void Conv3DBpInputTilingBase::SetFinalTiling(optiling::Conv3DBackpropInputTilingData& tiling)
{
    tiling.conv3DDxTiling.set_batch(this->shapeInfo.orgN);
    tiling.conv3DDxTiling.set_cin(this->shapeInfo.orgCi);
    tiling.conv3DDxTiling.set_cin1(this->shapeCalc.Ci1);
    tiling.conv3DDxTiling.set_di(this->shapeInfo.orgDi);
    tiling.conv3DDxTiling.set_hi(this->shapeInfo.orgHi);
    tiling.conv3DDxTiling.set_wi(this->shapeInfo.orgWi);
    tiling.conv3DDxTiling.set_dk(this->shapeInfo.orgkD);
    tiling.conv3DDxTiling.set_hk(this->shapeInfo.orgkH);
    tiling.conv3DDxTiling.set_wk(this->shapeInfo.orgkW);
    tiling.conv3DDxTiling.set_cout(this->shapeInfo.orgCo);
    tiling.conv3DDxTiling.set_cout1(this->shapeCalc.Co1);
    tiling.conv3DDxTiling.set_dout(this->shapeInfo.orgDo);
    tiling.conv3DDxTiling.set_ho(this->shapeInfo.orgHo);
    tiling.conv3DDxTiling.set_wo(this->shapeInfo.orgWo);
    // group param
    tiling.conv3DDxTiling.set_cin1G(this->shapeCalc.cin1G);
    tiling.conv3DDxTiling.set_cout1G(this->shapeCalc.cout1G);

    tiling.conv3DDxTiling.set_group(this->shapeCalc.realG);
    tiling.conv3DDxTiling.set_padFront(this->attrInfo.padFront);
    tiling.conv3DDxTiling.set_padBack(this->attrInfo.padBack);
    tiling.conv3DDxTiling.set_padUp(this->attrInfo.padUp);
    tiling.conv3DDxTiling.set_padDown(this->attrInfo.padDown);
    tiling.conv3DDxTiling.set_padLeft(this->attrInfo.padLeft);
    tiling.conv3DDxTiling.set_padRight(this->attrInfo.padRight);
    tiling.conv3DDxTiling.set_backpropPadTail(this->attrInfo.backpropPadTail);
    tiling.conv3DDxTiling.set_backpropPadUp(this->attrInfo.backpropPadUp);
    tiling.conv3DDxTiling.set_backpropPadDown(this->attrInfo.backpropPadDown);
    tiling.conv3DDxTiling.set_backpropPadLeft(this->attrInfo.backpropPadLeft);
    tiling.conv3DDxTiling.set_backpropPadRight(this->attrInfo.backpropPadRight);
    tiling.conv3DDxTiling.set_strideH(this->attrInfo.strideH);
    tiling.conv3DDxTiling.set_strideW(this->attrInfo.strideW);
    tiling.conv3DDxTiling.set_strideD(this->attrInfo.strideD);
    tiling.conv3DDxTiling.set_dilationH(this->attrInfo.dilationH);
    tiling.conv3DDxTiling.set_dilationW(this->attrInfo.dilationW);
    tiling.conv3DDxTiling.set_dilationD(this->attrInfo.dilationD);
    tiling.conv3DDxTiling.set_hf32Flag(this->attrInfo.hf32Enable);
}

void Conv3DBpInputTilingBase::SetFinalTiling(AscendC::tiling::Conv3DBackpropInputTilingData& tiling)
{
    optiling::Conv3DBackpropInputTilingData tilingData;
    SetFinalTiling(tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(Conv3DBackpropInputTilingData));
}
} // ConvBackpropApi
