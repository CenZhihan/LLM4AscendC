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
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief AscendC l311 support vector ternary scalar api.
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/basic_api/dav_l311/kernel_operator_vec_ternary_scalar_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_vec_intf.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_operator_common_impl.h"
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T>
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue,
                                          uint64_t mask[2], const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        MaskReg preg = MovePredicate<T>();
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);
        uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);
        uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);
        uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, src, srcBlkStride, srcRepStride, preg);
            DataCopy(vreg1, dst, dstBlkStride, 0, preg);
            Axpy(vreg1, vreg0, scalarValue, preg);
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, dstBlkStride, dstRepStride, preg);
        }
    }
}

template <typename T>
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue,
                                          uint64_t mask, const uint8_t repeatTime,
                                          const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        MaskReg preg;
        uint32_t sreg = (uint32_t)mask;
        preg = CreatePredicate<T>(sreg);
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);
        uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);
        uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);
        uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, src, srcBlkStride, srcRepStride, preg);
            DataCopy(vreg1, dst, dstBlkStride, 0, preg);
            Axpy(vreg1, vreg0, scalarValue, preg);
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(dst, vreg1, dstBlkStride, dstRepStride, preg);
        }
    }
}

template <typename T>
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue,
                                          const int32_t& count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1;
        MaskReg preg;
        uint32_t sreg = (uint32_t)count;
        constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            preg = CreatePredicate<T>(sreg);
            DataCopy(vreg0, src, i * sregLower);
            DataCopy(vreg1, dst, i * sregLower);
            Axpy(vreg1, vreg0, scalarValue, preg);
            DataCopy(dst, vreg1, i * sregLower, preg);
        }
    }
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask[2], const uint8_t repeatTime,
                                    const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        MaskReg preg = MovePredicate<float>();
        RegTensor<half> src_vreg;
        RegTensor<half> tmp_vreg;
        RegTensor<half> zero_vreg;
        RegTensor<float> cvt_vreg;
        RegTensor<float> dst_vreg;
        MaskReg full_preg;
        uint32_t full_sreg = FULL_MASK_LEN;
        full_preg = CreatePredicate<half>(full_sreg);
        Duplicate(zero_vreg, (half)0, full_preg);
        uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);
        uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);
        uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);
        uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            DataCopy<half, PostLiteral::POST_MODE_UPDATE>(src_vreg, src, srcBlkStride, srcRepStride, preg);
            Interleave(src_vreg, tmp_vreg, src_vreg, zero_vreg);
            Cast<float, half, Mode::ZEROING, PartMode::EVEN>(cvt_vreg, src_vreg, preg);
            DataCopy(dst_vreg, dst, dstBlkStride, 0, preg);
            Axpy(dst_vreg, cvt_vreg, (float)scalarValue, preg);
            DataCopy<float, PostLiteral::POST_MODE_UPDATE>(dst, dst_vreg, dstBlkStride, dstRepStride, preg);
        }
    }
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask, const uint8_t repeatTime,
                                    const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        MaskReg preg;
        uint32_t sreg = (uint32_t)mask;
        preg = CreatePredicate<float>(sreg);
        RegTensor<half> src_vreg;
        RegTensor<half> tmp_vreg;
        RegTensor<half> zero_vreg;
        RegTensor<float> cvt_vreg;
        RegTensor<float> dst_vreg;
        MaskReg full_preg;
        uint32_t full_sreg = FULL_MASK_LEN;
        full_preg = CreatePredicate<half>(full_sreg);
        Duplicate(zero_vreg, (half)0, full_preg);
        uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);
        uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);
        uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);
        uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            DataCopy<half, PostLiteral::POST_MODE_UPDATE>(src_vreg, src, srcBlkStride, srcRepStride, preg);
            Interleave(src_vreg, tmp_vreg, src_vreg, zero_vreg);
            Cast<float, half, Mode::ZEROING, PartMode::EVEN>(cvt_vreg, src_vreg, preg);
            DataCopy(dst_vreg, dst, dstBlkStride, 0, preg);
            Axpy(dst_vreg, cvt_vreg, (float)scalarValue, preg);
            DataCopy<float, PostLiteral::POST_MODE_UPDATE>(dst, dst_vreg, dstBlkStride, dstRepStride, preg);
        }
    }
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    const int32_t& count)
{
    __VEC_SCOPE__ {
        RegTensor<half> src_vreg;
        RegTensor<half> tmp_vreg;
        RegTensor<half> zero_vreg;
        RegTensor<float> cvt_vreg;
        RegTensor<float> dst_vreg;
        MaskReg preg;
        MaskReg full_preg;
        uint32_t full_sreg = FULL_MASK_LEN;
        full_preg = CreatePredicate<half>(full_sreg);
        Duplicate(zero_vreg, (half)0, full_preg);
        uint32_t sreg = (uint32_t)count;
        uint16_t repeatTime = CeilDivision(count, B32_DATA_NUM_PER_REPEAT);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            preg = CreatePredicate<float>(sreg);
            DataCopy(src_vreg, src, i * B32_DATA_NUM_PER_REPEAT);
            Interleave(src_vreg, tmp_vreg, src_vreg, zero_vreg);
            Cast<float, half, Mode::ZEROING, PartMode::EVEN>(cvt_vreg, src_vreg, preg);
            DataCopy(dst_vreg, dst, i * B32_DATA_NUM_PER_REPEAT);
            Axpy(dst_vreg, cvt_vreg, (float)scalarValue, preg);
            DataCopy(dst, dst_vreg, i * B32_DATA_NUM_PER_REPEAT, preg);
        }
    }
}

// Axpy::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask[2], const uint8_t repeatTime,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    if constexpr ((sizeof(T) == sizeof(U)) && (std::is_same_v<T, half> || std::is_same_v<T, float>)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, half>) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask, const uint8_t repeatTime,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr ((sizeof(T) == sizeof(U)) && (std::is_same_v<T, half> || std::is_same_v<T, float>)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, half>) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTime, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

// Axpy::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                const int32_t& count)
{
    if constexpr ((sizeof(T) == sizeof(U)) && (std::is_same_v<T, half> || std::is_same_v<T, float>)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, count);
    } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, half>) {
        return AxpyFmixImpl(dst, src, scalarValue, count);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H__
#endif
