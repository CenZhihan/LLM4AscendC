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
 * \file conv3d_tiling_base.cpp
 * \brief
 */

#include "../../detail/host_log.h"
#include "include/adv_api/conv/conv3d/conv3d_tiling_base.h"

namespace Conv3dTilingApi {

Conv3dTilingBase::Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
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

Conv3dTilingBase::Conv3dTilingBase(const PlatformInfo& platform)
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

void Conv3dTilingBase::SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw)
{
    this->shapeInfo.orgkH = orgKh;
    this->shapeInfo.orgkW = orgKw;
    this->shapeInfo.orgkD = orgKd;
    this->shapeInfo.orgCo = orgCo;
    this->shapeInfo.coutOpt = this->shapeInfo.orgCo;
}

void Conv3dTilingBase::SetSingleWeightShape(int64_t singleCi, int64_t singleKd,
    int64_t singleKh, int64_t singleKw)
{
    this->shapeInfo.singlekH = singleKh;
    this->shapeInfo.singlekW = singleKw;
    this->shapeInfo.singlekD = singleKd;
    this->shapeInfo.singleCi = singleCi;
}

void Conv3dTilingBase::SetOrgInputShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
{
    this->shapeInfo.orgCi = orgCi;
    this->shapeInfo.orgHi = orgHi;
    this->shapeInfo.orgWi = orgWi;
    this->shapeInfo.orgDi = orgDi;
    this->shapeInfo.cinOpt = this->shapeInfo.orgCi;
}

void Conv3dTilingBase::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleM = singleM;
}

void Conv3dTilingBase::SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo)
{
    this->shapeInfo.singleCo = singleCo;
    this->shapeInfo.singleDo = singleDo;
    this->shapeInfo.singleHo = singleHo;
    this->shapeInfo.singleWo = singleWo;
    this->shapeInfo.singleM = singleHo * singleWo;
}

void Conv3dTilingBase::SetWeightType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.weightType.pos = pos;
    this->descInfo.weightType.format = format;
    this->descInfo.weightType.dtype = dtype;
}

void Conv3dTilingBase::SetInputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.fMapType.pos = pos;
    this->descInfo.fMapType.format = format;
    this->descInfo.fMapType.dtype = dtype;
}

void Conv3dTilingBase::SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown,
        int64_t padLeft, int64_t padRight)
{
    this->attrInfo.padHead = padHead;
    this->attrInfo.padTail = padTail;
    this->attrInfo.padUp = padUp;
    this->attrInfo.padDown = padDown;
    this->attrInfo.padLeft = padLeft;
    this->attrInfo.padRight = padRight;
}

void Conv3dTilingBase::SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
{
    this->attrInfo.dilationH = dilationH;
    this->attrInfo.dilationW = dilationW;
    this->attrInfo.dilationD = dilationD;
}

void Conv3dTilingBase::SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
{
    this->attrInfo.strideD = strideD;
    this->attrInfo.strideH = strideH;
    this->attrInfo.strideW = strideW;
}

void Conv3dTilingBase::SetBiasType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
{
    this->hasBias = true;
    this->descInfo.biasType.pos = pos;
    this->descInfo.biasType.format = format;
    this->descInfo.biasType.dtype = dtype;
}

void Conv3dTilingBase::SetHF32(bool hf32Enable, bool hf32TransMode = false)
{
    this->hf32Enable_ = hf32Enable;
    this->attrInfo.hf32Enable = hf32Enable;
    this->hf32TransMode_ = hf32TransMode;
    this->attrInfo.hf32TransMode = hf32TransMode;
}

void Conv3dTilingBase::SetGroups(int64_t groups)
{
    // this should be called after SetOrgWeightShape and SetOrgInputShape
    this->attrInfo.groups = groups;
    this->attrInfo.groupOpt = this->attrInfo.groups;
}

void Conv3dTilingBase::SetOutputType(const ConvCommonApi::TPosition pos, const ConvCommonApi::ConvFormat format, const ConvCommonApi::ConvDtype dtype)
{
    this->descInfo.outputType.pos = pos;
    this->descInfo.outputType.format = format;
    this->descInfo.outputType.dtype = dtype;
}

bool Conv3dTilingBase::CheckSocVersion()
{
    if (this->platformInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        this->platformInfo.socVersion != platform_ascendc::SocVersion::ASCEND910_93) {
        std::string strSocVersion = "RESERVED_VERSION";
        auto iter = SOC_NAME_TAB.find(this->platformInfo.socVersion);
        if (iter != SOC_NAME_TAB.end()) {
            strSocVersion = iter->second;
        }
        TILING_LOG_ERROR("The %s is not supported. Only support Ascend910B/910_93.", strSocVersion.c_str());
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputParam()
{
    if (!CheckInputAttr() || !CheckInputShape() || !CheckInputFormat() || !CheckParamsDtype()) {
        return false;
    }

    if (!CheckPaddedInput()) {
        return false;
    }

    if (!CheckInstructionLimits()) {
        return false;
    }

    if (!CheckHF32()) {
        return false;
    }

    return true;
}

static void CheckInputAttrUpper(const Conv3DInputAttr& conv3DInputAttr)
{
    if (conv3DInputAttr.strideD > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("strideD (%ld) is out of range[1, %ld].", conv3DInputAttr.strideD, MAX_ONE_DIM_SIZE);
    }

    if (conv3DInputAttr.dilationD > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("dilationD (%ld) is out of range[1, %ld].", conv3DInputAttr.dilationD, MAX_ONE_DIM_SIZE);
    }

    if (conv3DInputAttr.padHead > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("padHead (%ld) is out of range[0, %ld].", conv3DInputAttr.padHead, MAX_ONE_DIM_SIZE);
    }

    if (conv3DInputAttr.padTail > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("padTail (%ld) is out of range[0, %ld].", conv3DInputAttr.padTail, MAX_ONE_DIM_SIZE);
    }
}

bool Conv3dTilingBase::CheckInputAttr()
{
    if (this->attrInfo.groups != 1 || this->attrInfo.groupOpt != 1) {
        TILING_LOG_ERROR(
            "Illegal attrs have set: groups=%ld, groupOpt=%ld.", 
            this->attrInfo.groups, this->attrInfo.groupOpt);
        return false;
    }

    bool padInvalidFlag = (this->attrInfo.padLeft < 0 || this->attrInfo.padRight < 0 ||
        this->attrInfo.padUp < 0 || this->attrInfo.padDown < 0 ||
        this->attrInfo.padHead < 0 || this->attrInfo.padTail < 0);
    if (padInvalidFlag) {
        TILING_LOG_ERROR(
            "Illegal attrs have set: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld, padHead=%ld, padTail=%ld, "
            "which must >= 0.", this->attrInfo.padUp, this->attrInfo.padDown, this->attrInfo.padLeft,
            this->attrInfo.padRight, this->attrInfo.padHead, this->attrInfo.padTail);
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

    CheckInputAttrUpper(this->attrInfo);

    return true;
}

static void CheckOrgInputInfoUpper(const Conv3DInputshape& conv3DInputshape)
{
    if (conv3DInputshape.orgCi > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Cin (%ld) is out of range[1, %ld].", conv3DInputshape.orgCi, MAX_ONE_DIM_SIZE);
    }
    if (conv3DInputshape.orgDi > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Din (%ld) is out of range[1, %ld].", conv3DInputshape.orgDi, MAX_ONE_DIM_SIZE);
    }
    if (conv3DInputshape.orgHi > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Hin (%ld) is out of range[1, %ld].", conv3DInputshape.orgHi, MAX_ONE_DIM_SIZE);
    }
    if (conv3DInputshape.orgWi > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Win (%ld) is out of range[1, %ld].", conv3DInputshape.orgWi, MAX_ONE_DIM_SIZE);
    }

    if (conv3DInputshape.orgCo > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Cout (%ld) is out of range[1, %ld].", conv3DInputshape.orgCo, MAX_ONE_DIM_SIZE);
    }

    if (conv3DInputshape.orgkD > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("kD (%ld) is out of range[1, %ld].", conv3DInputshape.orgkD, MAX_ONE_DIM_SIZE);
    }

    int64_t inputSize = conv3DInputshape.orgCi * conv3DInputshape.orgDi * conv3DInputshape.orgHi * conv3DInputshape.orgWi;
    if (inputSize > MAX_ORI_INPUT_SIZE) {
        TILING_LOG_WARNING("Cin*Din*Hin*Win (%ld) is out of range[1, %ld].", inputSize, MAX_ORI_INPUT_SIZE);
    }
}

bool Conv3dTilingBase::CheckOrgInputInfo()
{
    if (this->shapeInfo.orgkH <= 0 || this->shapeInfo.orgkW <= 0 || 
        this->shapeInfo.orgCo <= 0 || this->shapeInfo.orgkD <= 0) {
        TILING_LOG_ERROR("Input illegal weight shapes have set: Co=%ld, kH=%ld, kW=%ld, kD=%ld, "
            "which must > 0.", this->shapeInfo.orgCo, this->shapeInfo.orgkH, 
            this->shapeInfo.orgkW, this->shapeInfo.orgkD);
        return false;
    }

    if (this->shapeInfo.orgCi <= 0 || this->shapeInfo.orgHi <= 0 || this->shapeInfo.orgWi <= 0 || this->shapeInfo.orgDi <= 0) {
        TILING_LOG_ERROR("Input illegal featureMap shapes have set: Ci=%ld, Hi=%ld, Wi=%ld, Di=%ld, which must > 0.",
            this->shapeInfo.orgCi, this->shapeInfo.orgHi, this->shapeInfo.orgWi, this->shapeInfo.orgDi);
        return false;
    }

    CheckOrgInputInfoUpper(this->shapeInfo);

    return true;
}

bool Conv3dTilingBase::CheckSingleInputInfo()
{
    if (this->shapeInfo.singlekH <= 0 || this->shapeInfo.singlekW <= 0 ||
        this->shapeInfo.singleCo <= 0 || this->shapeInfo.singlekD <= 0) {
        TILING_LOG_ERROR("Illegal singleCore weight shapes have set: Co=%ld, kH=%ld, kW=%ld, kD=%ld, which must > 0.",
            this->shapeInfo.singleCo, this->shapeInfo.singlekH, this->shapeInfo.singlekW, this->shapeInfo.singlekD);
        return false;
    }

    if (this->shapeInfo.singleCo > this->shapeInfo.orgCo) {
        TILING_LOG_ERROR("Illegal singleCo have set: singleCo=%ld, orgCo=%ld, which must <= orgCo.",
            this->shapeInfo.singleCo, this->shapeInfo.orgCo);
        return false;
    }

    if (this->shapeInfo.singleCi > this->shapeInfo.orgCi) {
        TILING_LOG_ERROR("Illegal singleCi have set: singleCi=%ld, orgCi=%ld, which must <= orgCi.",
            this->shapeInfo.singleCi, this->shapeInfo.orgCi);
        return false;
    }

    if (this->shapeInfo.singleCi <= 0 || this->shapeInfo.singleDo <= 0 || this->shapeInfo.singleM <= 0) {
        TILING_LOG_ERROR("Illegal singleCore featureMap shapes have set: Ci=%ld, Do=%ld, M=%ld, which must > 0.",
            this->shapeInfo.singleCi, this->shapeInfo.singleDo, this->shapeInfo.singleM);
        return false;
    }

    if (this->shapeInfo.cinOpt <= 0 || this->shapeInfo.coutOpt <= 0) {
        TILING_LOG_ERROR("Illegal cinOpt and coutOpt have set: cinOpt=%ld, coutOpt=%ld.",
            this->shapeInfo.cinOpt, this->shapeInfo.coutOpt);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputConstraint()
{
    if (this->shapeInfo.singlekH != this->shapeInfo.orgkH || this->shapeInfo.singlekW != this->shapeInfo.orgkW
        || this->shapeInfo.singlekD != this->shapeInfo.orgkD) {
        TILING_LOG_ERROR("Support singlekH = orgkH, singlekW = orgkW, singlekD = orgkD, "
            "current singlekH: %ld, orgkH: %ld, singlekW: %ld, orgkW: %ld, singlekD: %ld, orgkD: %ld", this->shapeInfo.singlekH,
            this->shapeInfo.orgkH, this->shapeInfo.singlekW, this->shapeInfo.orgkW, this->shapeInfo.singlekD,
            this->shapeInfo.orgkD);
        return false;
    }

    if (this->attrInfo.groups == 1 && (this->shapeInfo.orgCi != this->shapeInfo.cinOpt
        || this->shapeInfo.orgCo != this->shapeInfo.coutOpt)) {
        TILING_LOG_ERROR("cinOpt not equal to orgCi or coutOpt not equal to orgCo in groups 1");
        return false;
    }

    if (this->shapeInfo.singleCi != this->shapeInfo.cinOpt) {
        TILING_LOG_ERROR("Support singleCi = cinOpt, current singleCi: %ld, cinOpt: %ld", this->shapeInfo.singleCi,
                         this->shapeInfo.cinOpt);
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckInputShape()
{
    if (!CheckOrgInputInfo()) {
        return false;
    }

    if (!CheckSingleInputInfo()) {
        return false;
    }

    if (!CheckInputConstraint()) {
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckInputFormat()
{
    if (this->descInfo.weightType.format != ConvCommonApi::ConvFormat::FRACTAL_Z_3D) {
        TILING_LOG_ERROR("Unsupported weight format: %s.",
                         g_formatToStr.at(this->descInfo.weightType.format).c_str());
        return false;
    }

    if (this->descInfo.fMapType.format != ConvCommonApi::ConvFormat::NDC1HWC0) {
        TILING_LOG_ERROR("Unsupported input format: %s.",
                         g_formatToStr.at(this->descInfo.fMapType.format).c_str());
        return false;
    }
    if (this->descInfo.outputType.format != ConvCommonApi::ConvFormat::NDC1HWC0) {
        TILING_LOG_ERROR("Unsupported output format: %s",
                         g_formatToStr.at(this->descInfo.outputType.format).c_str());
        return false;
    }
    if (this->hasBias && this->descInfo.biasType.format != ConvCommonApi::ConvFormat::ND) {
        TILING_LOG_ERROR("Unsupported bias format: %s",
                        g_formatToStr.at(this->descInfo.biasType.format).c_str());
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckParamsDtype()
{
    if (this->hasBias) {
        std::vector<ConvCommonApi::ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype,
            this->descInfo.biasType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_BIAS)) {
                return true;
            }
        }
        TILING_LOG_ERROR("Unsupported params data type [input, weight, bias, output]: [%s, %s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.biasType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    } else {
        std::vector<ConvCommonApi::ConvDtype> paramsType = {
            this->descInfo.fMapType.dtype, this->descInfo.weightType.dtype, this->descInfo.outputType.dtype
        };

        for (uint64_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); ++kindsId) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_NO_BIAS)) {
                return true;
            }
        }
        TILING_LOG_ERROR("Unsupported params data type [input, weight, output]: [%s, %s, %s].",
            g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.weightType.dtype).c_str(),
            g_dtypeToStr.at(this->descInfo.outputType.dtype).c_str());
        return false;
    }
    return false;
}

bool Conv3dTilingBase::CheckLoad3DLimits()
{
    if (static_cast<uint32_t>(this->attrInfo.strideH) > LOAD3D_MAX_STRIDE_H_W ||
        static_cast<uint32_t>(this->attrInfo.strideW) > LOAD3D_MAX_STRIDE_H_W) {
        TILING_LOG_ERROR(
            "Attrs not satisfy Load3D's limits: strideH=%ld, strideW=%ld, which must <= %u.",
            this->attrInfo.strideH, this->attrInfo.strideW, LOAD3D_MAX_STRIDE_H_W);
        return false;
    }

    if (static_cast<uint32_t>(this->attrInfo.dilationH) > LOAD3D_MAX_DILATION_H_W ||
        static_cast<uint32_t>(this->attrInfo.dilationW) > LOAD3D_MAX_DILATION_H_W) {
        TILING_LOG_ERROR(
            "Attrs not satisfy Load3D's limits: dilationH=%ld, dilationW=%ld, which must <= %u.",
            this->attrInfo.dilationH, this->attrInfo.dilationW, LOAD3D_MAX_DILATION_H_W);
        return false;
    }

    bool padLoad3dInvalid = (static_cast<uint32_t>(this->attrInfo.padLeft) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padRight) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padUp) > LOAD3D_MAX_PAD ||
        static_cast<uint32_t>(this->attrInfo.padDown) > LOAD3D_MAX_PAD);
    if (padLoad3dInvalid) {
        TILING_LOG_ERROR(
            "Attrs not satisfy Load3D's limits: padUp=%ld, padDown=%ld, padLeft=%ld, padRight=%ld, "
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

    auto k0Size = C0_BYTE_SIZE / g_dtypeSizeTab.at(this->descInfo.fMapType.dtype);
    uint64_t tmpkHWSize = this->shapeInfo.orgkH * this->shapeInfo.orgkW * k0Size;
    if (tmpkHWSize > LOAD3D_MAX_DDR2L1_SIZE) {
        TILING_LOG_ERROR("Weight shape not satisfy Load3D's limits: kH*kW*k0=%lu, which must <= %u.",
                         tmpkHWSize, LOAD3D_MAX_DDR2L1_SIZE);
        return false;
    }

    return true;
}

bool Conv3dTilingBase::CheckLoadL1SizeLimits()
{
    uint64_t m0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_M_INDEX);
    uint32_t k0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_K_INDEX);
    uint64_t n0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_N_INDEX);
    uint32_t fMapDtypeSize = static_cast<uint32_t>(g_dtypeSizeTab.at(this->descInfo.fMapType.dtype));
    uint32_t biasDtypeSize = static_cast<uint32_t>(g_dtypeSizeTab.at(this->descInfo.biasType.dtype));

    uint32_t minBiasSize = this->hasBias ? AlignB(n0 * biasDtypeSize, C0_BYTE_SIZE): 0;
    uint64_t minAL1Size = 0;
    uint64_t orgWo = (this->shapeInfo.orgWi + this->attrInfo.padLeft + this->attrInfo.padRight -
                             this->attrInfo.dilationW * (this->shapeInfo.orgkW - 1) - 1) / this->attrInfo.strideW + 1;
    uint64_t hoAL1min = m0 / orgWo + 2;
    uint64_t tmpHiAL1 = InferHiL1(
        hoAL1min, this->shapeInfo.orgHi, this->shapeInfo.orgkH, this->attrInfo.dilationH, this->attrInfo.strideH);
    minAL1Size = tmpHiAL1 * this->shapeInfo.orgWi * k0 * fMapDtypeSize;

    uint64_t minL1LoadSize = minBiasSize + minAL1Size;
    if (minL1LoadSize > L1_SIZE) {
        TILING_LOG_ERROR("MinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %u", minL1LoadSize, L1_SIZE);
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckInstructionLimits()
{
    if (!CheckLoad3DLimits() || !CheckLoadL1SizeLimits()) {
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckHF32()
{
    if (this->hf32TransMode_) {
        TILING_LOG_ERROR("hf32TransMode only support false.");
        return false;
    }
    if (this->hf32Enable_ && this->descInfo.fMapType.dtype != ConvCommonApi::ConvDtype::FLOAT32) {
        TILING_LOG_ERROR("hf32Enable can only be true when the dtype is float32, but now is %s.",
                          g_dtypeToStr.at(this->descInfo.fMapType.dtype).c_str());
        return false;
    }
    return true;
}

bool Conv3dTilingBase::CheckPaddedInput()
{
    // calculate kernel size after dilation
    int64_t dilatedKernelD = (this->shapeInfo.orgkD - 1) * this->attrInfo.dilationD + 1;
    int64_t dilatedKernelH = (this->shapeInfo.orgkH - 1) * this->attrInfo.dilationH + 1;
    int64_t dilatedKernelW = (this->shapeInfo.orgkW - 1) * this->attrInfo.dilationW + 1;
    // calculate input size after add padding
    int64_t paddedInputD = this->shapeInfo.orgDi + this->attrInfo.padHead + this->attrInfo.padTail;
    int64_t paddedInputH = this->shapeInfo.orgHi + this->attrInfo.padUp + this->attrInfo.padDown;
    int64_t paddedInputW = this->shapeInfo.orgWi + this->attrInfo.padLeft + this->attrInfo.padRight;

    if (dilatedKernelD > paddedInputD) {
        TILING_LOG_ERROR("Dilated kernel depth[%ld] should not be larger than the padded input depth[%ld].",
                          dilatedKernelD, paddedInputD);
        return false;
    }
    if (dilatedKernelH > paddedInputH) {
        TILING_LOG_ERROR("Dilated kernel height[%ld] should not be larger than the padded input height[%ld].",
                          dilatedKernelH, paddedInputH);
        return false;
    }
    if (dilatedKernelW > paddedInputW) {
        TILING_LOG_ERROR("Dilated kernel width[%ld] should not be larger than the padded input width[%ld].",
                          dilatedKernelW, paddedInputW);
        return false;
    }

    return true;
}

void Conv3dTilingBase::SetFinalTilingBasicInfo(optiling::TConv3DApiTiling& tiling)
{
    // set origin shape info
    tiling.set_groups(this->attrInfo.groups);
    tiling.set_groupOpt(this->attrInfo.groupOpt);
    tiling.set_orgDo(this->shapeCalc.orgDo);
    tiling.set_orgCo(this->shapeInfo.orgCo);
    tiling.set_coutOpt(this->shapeInfo.coutOpt);
    tiling.set_orgHo(this->shapeCalc.orgHo);
    tiling.set_orgWo(this->shapeCalc.orgWo);
    tiling.set_orgCi(this->shapeInfo.orgCi);
    tiling.set_cinOpt(this->shapeInfo.cinOpt);
    tiling.set_orgDi(this->shapeInfo.orgDi);
    tiling.set_orgHi(this->shapeInfo.orgHi);
    tiling.set_orgWi(this->shapeInfo.orgWi);
    tiling.set_kernelD(this->shapeInfo.orgkD);
    tiling.set_kernelH(this->shapeInfo.orgkH);
    tiling.set_kernelW(this->shapeInfo.orgkW);
    tiling.set_orgHixWi(this->shapeInfo.orgHi * this->shapeInfo.orgWi);
    tiling.set_orgHoxWo(this->shapeCalc.orgHo * this->shapeCalc.orgWo);
    tiling.set_cin1xOriHixOriWixk0(this->shapeCalc.singleCi1 * this->shapeInfo.orgHi * this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_oriHixOriWixk0(this->shapeInfo.orgHi * this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_oriWixk0(this->shapeInfo.orgWi * this->cubeInfo.k0);
    tiling.set_kernelHxkernelW(this->shapeInfo.orgkH * this->shapeInfo.orgkW);
    // set single core info
    tiling.set_singleCoreCo(this->shapeInfo.singleCo);
    tiling.set_singleCoreDo(this->shapeInfo.singleDo);
    tiling.set_singleCoreM(this->shapeInfo.singleM);
    tiling.set_singleCoreGroupOpt(this->shapeInfo.singleCoreGroupOpt);
    // set conv attr
    tiling.set_strideH(this->attrInfo.strideH);
    tiling.set_strideW(this->attrInfo.strideW);
    tiling.set_strideD(this->attrInfo.strideD);
    tiling.set_dilationH(this->attrInfo.dilationH);
    tiling.set_dilationW(this->attrInfo.dilationW);
    tiling.set_dilationD(this->attrInfo.dilationD);
    tiling.set_padHead(this->attrInfo.padHead);
    tiling.set_padTail(this->attrInfo.padTail);
    tiling.set_padUp(this->attrInfo.padUp);
    tiling.set_padDown(this->attrInfo.padDown);
    tiling.set_padLeft(this->attrInfo.padLeft);
    tiling.set_padRight(this->attrInfo.padRight);
    tiling.set_offsetx(this->attrInfo.offsetx);
}

void Conv3dTilingBase::SetFinalTilingBasicInfo(AscendC::tiling::TConv3DApiTiling& tiling)
{
    optiling::TConv3DApiTiling tConv3DApiTiling;
    SetFinalTilingBasicInfo(tConv3DApiTiling);
    tConv3DApiTiling.SaveToBuffer(&tiling, sizeof(TConv3DApiTiling));
}

void Conv3dTilingBase::SetFinalTilingDecisionInfo(optiling::TConv3DApiTiling& tiling)
{
    // set L1 tiling decision
    tiling.set_kAL1(this->l1TilingInfo.kAL1);
    tiling.set_kBL1(this->l1TilingInfo.kBL1);
    tiling.set_mAL1(this->l1TilingInfo.mAL1Value);
    tiling.set_nBL1(this->l1TilingInfo.nBL1Value);
    tiling.set_mAL1DivmL0(this->l1TilingInfo.mAL1DivmL0);
    tiling.set_nBL1DivnL0(this->l1TilingInfo.nBL1DivnL0);
    tiling.set_cin1InAL1(this->l1TilingInfo.cin1InAL1);
    tiling.set_kAL1Tail(this->l1TilingInfo.kAL1Tail);
    tiling.set_cin1InAL1Tail(this->l1TilingInfo.cin1InAL1Tail);
    tiling.set_kBL1DivK0(this->l1TilingInfo.kBL1DivK0);
    tiling.set_kBL1Tail(this->l1TilingInfo.kBL1Tail);
    tiling.set_kBL1TailDivK0(this->l1TilingInfo.kBL1TailDivK0);
    tiling.set_iterateMNOrder(static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    tiling.set_biasFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    tiling.set_fixpParamsFullLoadFlag(static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    tiling.set_bl1BypassFlag(static_cast<uint8_t>(this->l1TilingInfo.isWeightBypass));
    tiling.set_al1FullLoad(static_cast<uint8_t>(this->l1TilingInfo.al1FullLoad));
    tiling.set_bl1FullLoad(static_cast<uint8_t>(this->l1TilingInfo.bl1FullLoad));

    // set L0 tiling decision
    tiling.set_mL0(this->l0TilingInfo.mL0);
    tiling.set_kL0(this->l0TilingInfo.kL0);
    tiling.set_nL0(this->l0TilingInfo.nL0);
    tiling.set_nL0xk0(this->l0TilingInfo.nL0xk0);
    tiling.set_kL0xorgCoAlignN0(this->l0TilingInfo.kL0xorgCoAlignN0);
    // set pb
    tiling.set_pBufferFlag(this->dbValue.pBufferFlag);
    // set hf32 mode
    tiling.set_hf32Enable(this->attrInfo.hf32Enable);
    tiling.set_hf32TransMode(0);
}

void Conv3dTilingBase::SetFinalTilingDecisionInfo(AscendC::tiling::TConv3DApiTiling& tiling)
{
    optiling::TConv3DApiTiling tConv3DApiTiling;
    SetFinalTilingDecisionInfo(tConv3DApiTiling);
    tConv3DApiTiling.SaveToBuffer(&tiling, sizeof(TConv3DApiTiling));
}

void Conv3dTilingBase::SetFinalTiling(optiling::TConv3DApiTiling& tiling)
{
    SetFinalTilingBasicInfo(tiling);
    SetFinalTilingDecisionInfo(tiling);
}

void Conv3dTilingBase::SetFinalTiling(AscendC::tiling::TConv3DApiTiling& tiling)
{
    SetFinalTilingBasicInfo(tiling);
    SetFinalTilingDecisionInfo(tiling);
}

void Conv3dTilingBase::PrintTilingDataBasicInfo() const
{
    TILING_LOG_DEBUG("groups: %ld", this->attrInfo.groups);
    TILING_LOG_DEBUG("groupOpt: %ld", this->attrInfo.groupOpt);
    TILING_LOG_DEBUG("orgDo: %lu", this->shapeCalc.orgDo);
    TILING_LOG_DEBUG("orgCo: %ld", this->shapeInfo.orgCo);
    TILING_LOG_DEBUG("coutOpt: %ld", this->shapeInfo.coutOpt);
    TILING_LOG_DEBUG("orgHo: %lu", this->shapeCalc.orgHo);
    TILING_LOG_DEBUG("orgWo: %lu", this->shapeCalc.orgWo);
    TILING_LOG_DEBUG("orgCi: %ld", this->shapeInfo.orgCi);
    TILING_LOG_DEBUG("cinOpt: %ld", this->shapeInfo.cinOpt);
    TILING_LOG_DEBUG("orgDi: %ld", this->shapeInfo.orgDi);
    TILING_LOG_DEBUG("orgHi: %ld", this->shapeInfo.orgHi);
    TILING_LOG_DEBUG("orgWi: %ld", this->shapeInfo.orgWi);
    TILING_LOG_DEBUG("orgkD: %ld", this->shapeInfo.orgkD);
    TILING_LOG_DEBUG("orgkH: %ld", this->shapeInfo.orgkH);
    TILING_LOG_DEBUG("orgkW: %ld", this->shapeInfo.orgkW);
    TILING_LOG_DEBUG("singleCo: %ld", this->shapeInfo.singleCo);
    TILING_LOG_DEBUG("singleDo: %ld", this->shapeInfo.singleDo);
    TILING_LOG_DEBUG("singleM: %ld", this->shapeInfo.singleM);
    TILING_LOG_DEBUG("singleCoreGroupOpt: %ld", this->shapeInfo.singleCoreGroupOpt);
    TILING_LOG_DEBUG("strideH: %ld", this->attrInfo.strideH);
    TILING_LOG_DEBUG("strideW: %ld", this->attrInfo.strideW);
    TILING_LOG_DEBUG("strideD: %ld", this->attrInfo.strideD);
    TILING_LOG_DEBUG("dilationH: %ld", this->attrInfo.dilationH);
    TILING_LOG_DEBUG("dilationW: %ld", this->attrInfo.dilationW);
    TILING_LOG_DEBUG("dilationD: %ld", this->attrInfo.dilationD);
    TILING_LOG_DEBUG("padHead: %ld", this->attrInfo.padHead);
    TILING_LOG_DEBUG("padTail: %ld", this->attrInfo.padTail);
    TILING_LOG_DEBUG("padUp: %ld", this->attrInfo.padUp);
    TILING_LOG_DEBUG("padDown: %ld", this->attrInfo.padDown);
    TILING_LOG_DEBUG("padLeft: %ld", this->attrInfo.padLeft);
    TILING_LOG_DEBUG("padRight: %ld", this->attrInfo.padRight);
    TILING_LOG_DEBUG("mAL1DivmL0: %lu", this->l1TilingInfo.mAL1DivmL0);
    TILING_LOG_DEBUG("nBL1DivnL0: %lu", this->l1TilingInfo.nBL1DivnL0);
    TILING_LOG_DEBUG("cin1InAL1: %lu", this->l1TilingInfo.cin1InAL1);
    TILING_LOG_DEBUG("kAL1Tail: %lu", this->l1TilingInfo.kAL1Tail);
    TILING_LOG_DEBUG("cin1InAL1Tail: %lu", this->l1TilingInfo.cin1InAL1Tail);
    TILING_LOG_DEBUG("kBL1DivK0: %lu", this->l1TilingInfo.kBL1DivK0);
    TILING_LOG_DEBUG("kBL1Tail: %lu", this->l1TilingInfo.kBL1Tail);
    TILING_LOG_DEBUG("kBL1TailDivK0: %lu", this->l1TilingInfo.kBL1TailDivK0);
    TILING_LOG_DEBUG("nL0xk0: %lu", this->l0TilingInfo.nL0xk0);
    TILING_LOG_DEBUG("kL0xorgCoAlignN0: %lu", this->l0TilingInfo.kL0xorgCoAlignN0);
    TILING_LOG_DEBUG("offsetx: %d", this->attrInfo.offsetx);
}

void Conv3dTilingBase::PrintTilingDataDecision() const
{
    TILING_LOG_DEBUG("kAL1: %lu", this->l1TilingInfo.kAL1);
    TILING_LOG_DEBUG("kBL1: %lu", this->l1TilingInfo.kBL1);
    TILING_LOG_DEBUG("mAL1Value: %lu", this->l1TilingInfo.mAL1Value);
    TILING_LOG_DEBUG("nBL1Value: %lu", this->l1TilingInfo.nBL1Value);
    TILING_LOG_DEBUG("iterateMNOrder: %u", static_cast<uint8_t>(this->l1TilingInfo.iterateMNOrder));
    TILING_LOG_DEBUG("biasFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.biasFullLoadFlag));
    TILING_LOG_DEBUG("fixpParamsFullLoadFlag: %u", static_cast<uint8_t>(this->l1TilingInfo.fixpParamsFullLoadFlag));
    TILING_LOG_DEBUG("isWeightBypass: %u", static_cast<uint8_t>(this->l1TilingInfo.isWeightBypass));
    TILING_LOG_DEBUG("al1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.al1FullLoad));
    TILING_LOG_DEBUG("bl1FullLoad: %u", static_cast<uint8_t>(this->l1TilingInfo.bl1FullLoad));
    TILING_LOG_DEBUG("mL0: %lu", this->l0TilingInfo.mL0);
    TILING_LOG_DEBUG("kL0: %lu", this->l0TilingInfo.kL0);
    TILING_LOG_DEBUG("nL0: %lu", this->l0TilingInfo.nL0);
    TILING_LOG_DEBUG("pBufferFlag: %lu", this->dbValue.pBufferFlag);
    TILING_LOG_DEBUG("hf32Enable: %u", this->attrInfo.hf32Enable);
    TILING_LOG_DEBUG("hf32TransMode: %u", this->attrInfo.hf32TransMode);
}
void Conv3dTilingBase::PrintTilingData() const
{
    PrintTilingDataBasicInfo();
    PrintTilingDataDecision();
}

void Conv3dTilingBase::GetCubeInfo()
{
    this->cubeInfo.m0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_M_INDEX);
    this->cubeInfo.k0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_K_INDEX);
    this->cubeInfo.n0 = CUBE_MKN_TAB.GetMKN(this->descInfo.fMapType.dtype, MKN_N_INDEX);
    this->cubeInfo.biasType = CUBE_TYPE_TAB.ToBiasType(this->descInfo.fMapType.dtype);
    this->cubeInfo.madType = CUBE_TYPE_TAB.ToMadType(this->descInfo.fMapType.dtype);
    this->cubeInfo.minBurstNum = MIN_BURST_SIZE / g_dtypeSizeTab.at(descInfo.fMapType.dtype);
}

bool Conv3dTilingBase::ShapeInitCalc()
{
    this->shapeCalc.orgHo = (this->shapeInfo.orgHi + this->attrInfo.padUp + this->attrInfo.padDown -
                             this->attrInfo.dilationH * (this->shapeInfo.orgkH - 1) - 1) / this->attrInfo.strideH + 1;
    this->shapeCalc.orgWo = (this->shapeInfo.orgWi + this->attrInfo.padLeft + this->attrInfo.padRight -
                             this->attrInfo.dilationW * (this->shapeInfo.orgkW - 1) - 1) / this->attrInfo.strideW + 1;
    this->shapeCalc.orgDo = (this->shapeInfo.orgDi + this->attrInfo.padHead + this->attrInfo.padTail -
                             this->attrInfo.dilationD * (this->shapeInfo.orgkD - 1) - 1) / this->attrInfo.strideD + 1;
    if (this->shapeCalc.orgHo <= 0 || this->shapeCalc.orgWo <= 0 || this->shapeCalc.orgDo <= 0) {
        TILING_LOG_ERROR("Illegal org output shapes cal get: Ho=%lu, Wo=%lu, Do=%lu, which should > 0.",
                         this->shapeCalc.orgHo, this->shapeCalc.orgWo, this->shapeCalc.orgDo);
        return false;
    }
    if (this->shapeCalc.orgDo > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Dout (%lu) is out of range[1, %lu].", this->shapeCalc.orgDo, MAX_OUT_ONE_DIM_SIZE);
    }
    if (this->shapeCalc.orgHo > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Hout (%lu) is out of range[1, %lu].", this->shapeCalc.orgHo, MAX_OUT_ONE_DIM_SIZE);
    }
    if (this->shapeCalc.orgWo > MAX_ONE_DIM_SIZE) {
        TILING_LOG_WARNING("Wout (%lu) is out of range[1, %lu].", this->shapeCalc.orgWo, MAX_OUT_ONE_DIM_SIZE);
    }
    if (this->shapeInfo.singleDo > static_cast<int64_t>(this->shapeCalc.orgDo)) {
        TILING_LOG_ERROR("Illegal singleDo have set: singleDo=%ld, orgDo=%lu, which should <= orgDo.",
                         this->shapeInfo.singleDo, this->shapeCalc.orgDo);
        return false;
    }
    this->shapeCalc.singleCi1 = CeilDiv(this->shapeInfo.singleCi, this->cubeInfo.k0);
    this->shapeCalc.singleCo1 = CeilDiv(this->shapeInfo.singleCo, this->cubeInfo.n0);
    this->shapeCalc.singleM1 = CeilDiv(this->shapeInfo.singleM, this->cubeInfo.m0);

    return true;
}

bool Conv3dTilingBase::CheckParamsOverflow()
{
    uint64_t prod;
    bool isOverflow = MulWithOverflowCheck(
        prod, static_cast<uint64_t>(this->shapeInfo.orgDi),
        static_cast<uint64_t>(this->attrInfo.groupOpt), CeilDiv(this->shapeInfo.cinOpt, this->cubeInfo.k0),
        static_cast<uint64_t>(this->shapeInfo.orgHi), static_cast<uint64_t>(this->shapeInfo.orgWi),
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.fMapType.dtype)
    ) || MulWithOverflowCheck(
        prod, static_cast<uint64_t>(this->attrInfo.groupOpt), static_cast<uint64_t>(this->shapeInfo.singlekD),
        CeilDiv(this->shapeInfo.cinOpt, this->cubeInfo.k0),
        static_cast<uint64_t>(this->shapeInfo.singlekH), static_cast<uint64_t>(this->shapeInfo.singlekW),
        CeilDiv(this->shapeInfo.coutOpt, this->cubeInfo.n0), static_cast<uint64_t>(this->cubeInfo.n0) *
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.weightType.dtype)
    ) || MulWithOverflowCheck(
        prod, this->shapeCalc.orgDo, static_cast<uint64_t>(this->attrInfo.groupOpt),
        CeilDiv(this->shapeInfo.coutOpt, this->cubeInfo.k0), this->shapeCalc.orgHo, this->shapeCalc.orgWo,
        static_cast<uint64_t>(this->cubeInfo.k0) * g_dtypeSizeTab.at(this->descInfo.outputType.dtype)
    );
    if (isOverflow) {
        TILING_LOG_ERROR("Overflow detected, input or weight size exceeds UINT64_MAX.");
        return false;
    }
    return true;
}

} // namespace Conv3dTilingApi
