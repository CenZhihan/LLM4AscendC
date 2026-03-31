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
 * \file conv3d_bp_filter_tiling_base.cpp
 * \brief
 */
#include "../../detail/host_log.h"
#include "conv3d_bp_tiling_util.h"
#include "include/adv_api/conv_backprop/conv3d_bp_filter_tiling_base.h"

namespace ConvBackpropApi {
constexpr uint64_t L1_SIZE_LOCAL = 524288;
constexpr uint64_t L0A_SIZE = 65536;
constexpr uint64_t L0B_SIZE = 65536;
constexpr uint64_t L0C_SIZE = 131072;
constexpr uint64_t UB_SIZE = 262144;
constexpr uint64_t BT_SIZE = 1024;
constexpr uint64_t FB_SIZE = 2048;
constexpr uint32_t MAX_KW_SIZE = 32;   // transdata limit, the current KW limit exceeds 38
constexpr int32_t BLOCK_CUBE = 16;
constexpr int64_t BLOCK_CUBE_S64 = 16;

Conv3dBpFilterTilingBase::Conv3dBpFilterTilingBase()
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

Conv3dBpFilterTilingBase::Conv3dBpFilterTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
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
    this->platformInfo.l1Size = l1Size;
    this->platformInfo.l0ASize = l0ASize;
    this->platformInfo.l0BSize = l0BSize;
    this->platformInfo.l0CSize = l0CSize;
    this->platformInfo.ubSize = ubSize;
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

Conv3dBpFilterTilingBase::Conv3dBpFilterTilingBase(const PlatformInfo& platform)
{
    this->platformInfo.socVersion = platform.socVersion;
    this->platformInfo.l1Size = platform.l1Size;
    this->platformInfo.l0ASize = platform.l0ASize;
    this->platformInfo.l0BSize = platform.l0BSize;
    this->platformInfo.l0CSize = platform.l0CSize;
    this->platformInfo.ubSize = platform.ubSize;
    this->platformInfo.btSize = BT_SIZE;
    this->platformInfo.fbSize = FB_SIZE;
}

void Conv3dBpFilterTilingBase::SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
{
    this->shapeInfo.orgCo = cout;
    this->shapeInfo.orgCi = cin;
    this->shapeInfo.orgkH = h;
    this->shapeInfo.orgkW = w;
    this->shapeInfo.orgkD = d;
}

void Conv3dBpFilterTilingBase::SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    this->shapeInfo.orgN = n;
    this->shapeInfo.orgCi = c;
    this->shapeInfo.orgHi = h;
    this->shapeInfo.orgWi = w;
    this->shapeInfo.orgDi = d;
}

void Conv3dBpFilterTilingBase::SetGradOutputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
{
    this->shapeInfo.orgN = n;
    this->shapeInfo.orgCo = c;
    this->shapeInfo.orgDo = d;
    this->shapeInfo.orgHo = h;
    this->shapeInfo.orgWo = w;
}

void Conv3dBpFilterTilingBase::SetWeightType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.weightType.pos = pos;
    this->descInfo.weightType.format = format;
    this->descInfo.weightType.dtype = dtype;
}

void Conv3dBpFilterTilingBase::SetInputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.fMapType.pos = pos;
    this->descInfo.fMapType.format = format;
    this->descInfo.fMapType.dtype = dtype;
}

void Conv3dBpFilterTilingBase::SetGradOutputType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.outBackpropType.pos = pos;
    this->descInfo.outBackpropType.format = format;
    this->descInfo.outBackpropType.dtype = dtype;
}


void Conv3dBpFilterTilingBase::SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight)
{
    this->attrInfo.padFront = padFront;
    this->attrInfo.padBack = padBack;
    this->attrInfo.padUp = padUp;
    this->attrInfo.padDown = padDown;
    this->attrInfo.padLeft = padLeft;
    this->attrInfo.padRight = padRight;
}

void Conv3dBpFilterTilingBase::SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
{
    this->attrInfo.dilationH = dilationH;
    this->attrInfo.dilationW = dilationW;
    this->attrInfo.dilationD = dilationD;
}

void Conv3dBpFilterTilingBase::SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
{
    this->attrInfo.strideH = strideH;
    this->attrInfo.strideW = strideW;
    this->attrInfo.strideD = strideD;
}

void Conv3dBpFilterTilingBase::SetHF32(bool hf32Enable)
{
    this->hf32Enable_ = hf32Enable;
    this->attrInfo.hf32Enable = hf32Enable;
}

void Conv3dBpFilterTilingBase::SetGroup(int64_t groups)
{
    // this should be called after SetOrgWeightShape and SetOrgFmapShape
    // only support groups = 1
    this->attrInfo.groups = groups;
}

bool Conv3dBpFilterTilingBase::ShapeInitCalc()
{
    dtypeByte_ = g_dtypeSizeTab.at(this->descInfo.fMapType.dtype);
    int64_t k0 = C0_BYTE_SIZE / dtypeByte_;
    shapeCalc.channelSize = k0;
    shapeCalc.Ci1 = CeilDivision(shapeInfo.orgCi, k0);
    shapeCalc.Co1 = CeilDivision(shapeInfo.orgCo, k0);

    int64_t magFactor0 = Lcm(shapeInfo.orgCi / attrInfo.groups, k0) / (shapeInfo.orgCi / attrInfo.groups);
    int64_t magFactor1 = Lcm(shapeInfo.orgCo / attrInfo.groups, BLOCK_CUBE) / (shapeInfo.orgCo / attrInfo.groups);
    int64_t magFactor = Min(Lcm(magFactor0, magFactor1), attrInfo.groups);
    shapeCalc.real_g = (attrInfo.groups + magFactor - 1) / magFactor;
    shapeCalc.cin1_g = CeilDivision(magFactor * shapeInfo.orgCi / attrInfo.groups, k0);
    shapeCalc.cout1_g = CeilAlign(magFactor * shapeInfo.orgCo / attrInfo.groups, BLOCK_CUBE_S64) / k0;
    return true;
}

void Conv3dBpFilterTilingBase::PrintTilingData() const
{
    TILING_LOG_DEBUG("Batch Size: %ld", shapeInfo.orgN);
    TILING_LOG_DEBUG("Input Channels: %ld", shapeInfo.orgCi);
    TILING_LOG_DEBUG("Depth Input: %ld", shapeInfo.orgDi);
    TILING_LOG_DEBUG("Height Input: %ld", shapeInfo.orgHi);
    TILING_LOG_DEBUG("Width Input: %ld", shapeInfo.orgWi);
    TILING_LOG_DEBUG("Depth Kernel: %ld", shapeInfo.orgkD);
    TILING_LOG_DEBUG("Height Kernel: %ld", shapeInfo.orgkH);
    TILING_LOG_DEBUG("Width Kernel: %ld", shapeInfo.orgkW);
    TILING_LOG_DEBUG("output channel:%ld", shapeInfo.orgCo);
    TILING_LOG_DEBUG("Depth output:%ld", shapeInfo.orgDo);
    TILING_LOG_DEBUG("Height output:%ld", shapeInfo.orgHo);
    TILING_LOG_DEBUG("width output:%ld", shapeInfo.orgWo);
    TILING_LOG_DEBUG("Input Channels per Group:%ld", shapeCalc.cin1_g);
    TILING_LOG_DEBUG("Output Channels per Group:%ld", shapeCalc.cout1_g);
    TILING_LOG_DEBUG("Channel Size:%ld", shapeCalc.channelSize);
    TILING_LOG_DEBUG("Number of Groups:%ld", attrInfo.groups);
    TILING_LOG_DEBUG("front pad : %ld", attrInfo.padFront);
    TILING_LOG_DEBUG("back pad: %ld", attrInfo.padBack);
    TILING_LOG_DEBUG("up pad: %ld", attrInfo.padUp);
    TILING_LOG_DEBUG("down pad: %ld", attrInfo.padDown);
    TILING_LOG_DEBUG("left pad: %ld", attrInfo.padLeft);
    TILING_LOG_DEBUG("right pad: %ld", attrInfo.padRight);
    TILING_LOG_DEBUG("stride height: %ld", attrInfo.strideH);
    TILING_LOG_DEBUG("stride width:%ld", attrInfo.strideW);
    TILING_LOG_DEBUG("stride depth:%ld", attrInfo.strideD);
    TILING_LOG_DEBUG("dilation height :%ld", attrInfo.dilationH);
    TILING_LOG_DEBUG("dilation width :%ld", attrInfo.dilationW);
    TILING_LOG_DEBUG("dilation depth :%ld", attrInfo.dilationD);
    TILING_LOG_DEBUG("hf32 Flag:%u", attrInfo.hf32Enable);
}

void Conv3dBpFilterTilingBase::SetFinalTiling(optiling::Conv3DBackpropFilterTilingData& tiling)
{
    tiling.dwTiling.set_batch(static_cast<uint32_t>(this->shapeInfo.orgN));
    tiling.dwTiling.set_cin(static_cast<uint32_t>(this->shapeInfo.orgCi));
    tiling.dwTiling.set_di(static_cast<uint32_t>(this->shapeInfo.orgDi));
    tiling.dwTiling.set_hi(static_cast<uint32_t>(this->shapeInfo.orgHi));
    tiling.dwTiling.set_wi(static_cast<uint32_t>(this->shapeInfo.orgWi));
    tiling.dwTiling.set_dk(static_cast<uint32_t>(this->shapeInfo.orgkD));
    tiling.dwTiling.set_hk(static_cast<uint32_t>(this->shapeInfo.orgkH));
    tiling.dwTiling.set_wk(static_cast<uint32_t>(this->shapeInfo.orgkW));
    tiling.dwTiling.set_cout(static_cast<uint32_t>(this->shapeInfo.orgCo));
    tiling.dwTiling.set_dout(static_cast<uint32_t>(this->shapeInfo.orgDo));
    tiling.dwTiling.set_ho(static_cast<uint32_t>(this->shapeInfo.orgHo));
    tiling.dwTiling.set_wo(static_cast<uint32_t>(this->shapeInfo.orgWo));
    tiling.dwTiling.set_cin1G(static_cast<uint32_t>(this->shapeCalc.cin1_g));
    tiling.dwTiling.set_cout1G(static_cast<uint32_t>(this->shapeCalc.cout1_g));
    tiling.dwTiling.set_channelSize(static_cast<uint32_t>(this->shapeCalc.channelSize));
    tiling.dwTiling.set_group(static_cast<uint32_t>(this->attrInfo.groups));
    tiling.dwTiling.set_padFront(static_cast<uint32_t>(this->attrInfo.padFront));
    tiling.dwTiling.set_padBack(static_cast<uint32_t>(this->attrInfo.padBack));
    tiling.dwTiling.set_padUp(static_cast<uint32_t>(this->attrInfo.padUp));
    tiling.dwTiling.set_padDown(static_cast<uint32_t>(this->attrInfo.padDown));
    tiling.dwTiling.set_padLeft(static_cast<uint32_t>(this->attrInfo.padLeft));
    tiling.dwTiling.set_padRight(static_cast<uint32_t>(this->attrInfo.padRight));
    tiling.dwTiling.set_strideH(static_cast<uint32_t>(this->attrInfo.strideH));
    tiling.dwTiling.set_strideW(static_cast<uint32_t>(this->attrInfo.strideW));
    tiling.dwTiling.set_strideD(static_cast<uint32_t>(this->attrInfo.strideD));
    tiling.dwTiling.set_dilationH(static_cast<uint32_t>(this->attrInfo.dilationH));
    tiling.dwTiling.set_dilationW(static_cast<uint32_t>(this->attrInfo.dilationW));
    tiling.dwTiling.set_dilationD(static_cast<uint32_t>(this->attrInfo.dilationD));
    tiling.dwTiling.set_hf32Flag(static_cast<uint32_t>(this->attrInfo.hf32Enable));
}

void Conv3dBpFilterTilingBase::SetFinalTiling(AscendC::tiling::Conv3DBackpropFilterTilingData& tiling)
{
    optiling::Conv3DBackpropFilterTilingData tilingData;
    SetFinalTiling(tilingData);
    tilingData.SaveToBuffer(&tiling, sizeof(Conv3DBackpropFilterTilingData));
}

bool Conv3dBpFilterTilingBase::CheckInputParam()
{
    if (!CheckInputAttr() || !CheckInputShape() || !CheckInputFormat()) {
        return false;
    }

    if (!CheckInstructionLimits()) {
        return false;
    }

    return true;
}

bool Conv3dBpFilterTilingBase::CheckInputAttr()
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

bool Conv3dBpFilterTilingBase::CheckInputShape()
{
    if (this->shapeInfo.orgkH <= 0 || this->shapeInfo.orgkW <= 0 ||
        this->shapeInfo.orgkD <= 0) {
        TILING_LOG_ERROR("Illegal org weight shapes have set: kH=%ld, kW=%ld, kD=%ld, which must > 0.",
            this->shapeInfo.orgkH, this->shapeInfo.orgkW, this->shapeInfo.orgkD);
        return false;
    }

    if (this->shapeInfo.orgkW > MAX_KW_SIZE) {
        TILING_LOG_ERROR("Illegal org weight kW have set: kW=%ld, which must <= %u.",
            this->shapeInfo.orgkW, MAX_KW_SIZE);
        return false;
    }

    if (this->shapeInfo.orgCi <= 0 || this->shapeInfo.orgHi <= 0 ||
        this->shapeInfo.orgWi <= 0 || this->shapeInfo.orgDi <= 0) {
        TILING_LOG_ERROR("Illegal org featureMap shapes have set: Ci=%ld, Hi=%ld, Wi=%ld, Di=%ld, which must > 0.",
            this->shapeInfo.orgCi, this->shapeInfo.orgHi, this->shapeInfo.orgWi, this->shapeInfo.orgDi);
        return false;
    }

    if (this->shapeInfo.orgCo <= 0 || this->shapeInfo.orgHo <= 0 ||
        this->shapeInfo.orgWo <= 0 || this->shapeInfo.orgDo <= 0) {
        TILING_LOG_ERROR("Illegal org featureMap shapes have set: Co=%ld, Ho=%ld, Wo=%ld, Do=%ld, which must > 0.",
            this->shapeInfo.orgCo, this->shapeInfo.orgHo, this->shapeInfo.orgWo, this->shapeInfo.orgDo);
        return false;
    }

    return true;
}

bool Conv3dBpFilterTilingBase::CheckInputFormat()
{
    if (this->descInfo.weightType.format != ConvCommonApi::ConvFormat::FRACTAL_Z_3D) {
        TILING_LOG_ERROR("Unsupported weight format: %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }

    if (this->descInfo.fMapType.format != ConvCommonApi::ConvFormat::NDC1HWC0) {
        TILING_LOG_ERROR("Unsupported feature map format: %s.",
                         g_formatToStr.at(this->descInfo.fMapType.format).c_str());
        return false;
    }

    return true;
}

bool Conv3dBpFilterTilingBase::CheckLoad3DLimits()
{
    if (static_cast<uint32_t>(this->attrInfo.strideH) > LOAD3D_MAX_STRIDE_H_W ||
        static_cast<uint32_t>(this->attrInfo.strideW) > LOAD3D_MAX_STRIDE_H_W) {
        TILING_LOG_ERROR(
            "Attrs not satisfying load3d's limits: strideH=%ld, strideW=%ld, which must <= %u.",
            this->attrInfo.strideH, this->attrInfo.strideW, LOAD3D_MAX_STRIDE_H_W);
        return false;
    }

    if (static_cast<uint32_t>(this->attrInfo.dilationH) > LOAD3D_MAX_DILATION_H_W ||
        static_cast<uint32_t>(this->attrInfo.dilationW) > LOAD3D_MAX_DILATION_H_W) {
        TILING_LOG_ERROR(
            "Attrs not satisfying load3d's limits: dilationH=%ld, dilationW=%ld, which must <= %u.",
            this->attrInfo.dilationH, this->attrInfo.dilationW, LOAD3D_MAX_DILATION_H_W);
        return false;
    }

    bool padLoad3dInvalid = (static_cast<uint32_t>(this->attrInfo.padLeft) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padRight) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padUp) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padDown) > LOAD3D_MAX_PAD);
    if (padLoad3dInvalid) {
        TILING_LOG_ERROR(
            "Attrs not satisfying load3d's limits: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld,"\
            "which must <= %u.",
            this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft, this->attrInfo.padRight,
            LOAD3D_MAX_PAD);
        return false;
    }

    if (this->shapeInfo.orgkH > LOAD3D_MAX_FILTER_H_W || this->shapeInfo.orgkW > LOAD3D_MAX_FILTER_H_W) {
        TILING_LOG_ERROR("Weight shape not satisfy Load3D's limits: kh=%ld, kw=%ld, which must <= %u.",
                         this->shapeInfo.orgkH, this->shapeInfo.orgkW, LOAD3D_MAX_FILTER_H_W);
        return false;
    }

    return true;
}

bool Conv3dBpFilterTilingBase::CheckInstructionLimits()
{
    if (!CheckLoad3DLimits()) {
        return false;
    }
    return true;
}

} // namespace ConvBackpropApi
