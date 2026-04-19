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
 * \file kernel_operator_vec_binary_impl.h
 * \brief AscendC l300 support vector binary api.
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/basic_api/dav_l300/kernel_operator_vec_binary_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_vec_intf.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_binary_continuous_impl.h"
#include "kernel_operator_vec_template_impl.h"

namespace AscendC {

namespace CastParam {
constexpr Reg::CastTrait mulAddDstTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
}

const uint32_t B64_DATA_NUM_PER_REPEAT = 32;
const uint32_t B4_BYTE_SIZE_PER_REPEAT = 64;
const uint32_t L1_DUMP_UB_SIZE = TOTAL_UB_SIZE - 32 * 1024;

// for Level 0 binary op
#define BINARY_OP_IMPL_NOT_SUPPORT(FUNC_NAME)                                                                      \
    template <typename T, bool isSetMask = true>                                                                   \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[2],  \
        const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)                                         \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }
// for Level 0 binary op
#define BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(FUNC_NAME)                                                      \
    template <typename T, bool isSetMask = true>                                                                   \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,     \
        const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)                                         \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }
/* **************************************************************************************************
 * bit mask                                         *
 * ************************************************************************************************* */
// Level 0
#define BINARY_OP_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                                        \
    template <typename T, bool isSetMask = true>                                                                             \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src0, __ubuf__ DATA_TYPE* src1,            \
        const uint64_t mask[2],                                                                                              \
        const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)                                                   \
    {                                                                                                                        \
        if constexpr (isSetMask) {                                                                                           \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                                      \
        }                                                                                                                    \
        __VEC_SCOPE__                                                                                                        \
        {                                                                                                                    \
            RegTensor<DATA_TYPE> vreg0;                                                                                      \
            RegTensor<DATA_TYPE> vreg1;                                                                                      \
            RegTensor<DATA_TYPE> vreg2;                                                                                      \
            MaskReg preg = MovePredicate<DATA_TYPE>();                                                                       \
            uint32_t strideConfig0 = static_cast<uint32_t>(repeatParams.src0BlkStride);                                      \
            uint32_t repeatStrideConfig0 = static_cast<uint32_t>(repeatParams.src0RepStride);                               \
            uint32_t strideConfig1 = static_cast<uint32_t>(repeatParams.src1BlkStride);                                     \
            uint32_t repeatStrideConfig1 = static_cast<uint32_t>(repeatParams.src1RepStride);                               \
            uint32_t strideConfig2 = static_cast<uint32_t>(repeatParams.dstBlkStride);                                      \
            uint32_t repeatStrideConfig2 = static_cast<uint32_t>(repeatParams.dstRepStride);                                 \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                                           \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src0, strideConfig0, repeatStrideConfig0, preg);    \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg1, src1, strideConfig1, repeatStrideConfig1, preg);    \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                          \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg2, strideConfig2, repeatStrideConfig2, preg);     \
            }                                                                                                                \
        }                                                                                                                    \
    }

/* **************************************************************************************************
 * continuous mask                                            *
 * ************************************************************************************************* */
// Level 0
#define BINARY_OP_CONTINUOUS_MASK_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                       \
    template <typename T, bool isSetMask = true>                                                                             \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src0, __ubuf__ DATA_TYPE* src1,           \
        const uint64_t mask,                                                                                                \
        const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)                                                  \
    {                                                                                                                       \
        __VEC_SCOPE__                                                                                                       \
        {                                                                                                                   \
            RegTensor<DATA_TYPE> vreg0;                                                                                     \
            RegTensor<DATA_TYPE> vreg1;                                                                                     \
            RegTensor<DATA_TYPE> vreg2;                                                                                     \
            uint32_t sreg = (uint32_t)mask;                                                                                 \
            MaskReg preg = CreatePredicate<DATA_TYPE>(sreg);                                                                \
            uint32_t strideConfig0 = static_cast<uint32_t>(repeatParams.src0BlkStride);                                                  \
            uint32_t repeatStrideConfig0 = static_cast<uint32_t>(repeatParams.src0RepStride);                                            \
            uint32_t strideConfig1 = static_cast<uint32_t>(repeatParams.src1BlkStride);                                                  \
            uint32_t repeatStrideConfig1 = static_cast<uint32_t>(repeatParams.src1RepStride);                                            \
            uint32_t strideConfig2 = static_cast<uint32_t>(repeatParams.dstBlkStride);                                                   \
            uint32_t repeatStrideConfig2 = static_cast<uint32_t>(repeatParams.dstRepStride);                                             \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                                          \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg0, src0, strideConfig0, repeatStrideConfig0, preg);   \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(vreg1, src1, strideConfig1, repeatStrideConfig1, preg);   \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, vreg2, strideConfig2, repeatStrideConfig2, preg);    \
            }                                                                                                               \
        }                                                                                                                   \
    }

/* **************************************************************************************************
 * Or                                                                                               *
 * **************************************************************************************************/
// Or::Level 0
BINARY_OP_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_IMPL(OrImpl, Or, int16_t)
BINARY_OP_IMPL(OrImpl, Or, uint16_t)
BINARY_OP_IMPL(OrImpl, Or, int32_t)
BINARY_OP_IMPL(OrImpl, Or, uint32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint8_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint16_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, int32_t)
BINARY_OP_CONTINUOUS_MASK_IMPL(OrImpl, Or, uint32_t)

// Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Sub<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Div<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Max<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::And<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Add<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Sub<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Mul<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, uint16_t, int16_t, uint32_t, int32_t, half, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Div<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Max<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, uint16_t, int16_t, uint32_t, int32_t, float>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::Min<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()),
        "current data type is not supported on current device!");
    constexpr auto func = Reg::And<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t &calCount)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)calCount;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
}

// AddDeqRelu::Level 0
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[2], const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTimes;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTimes;
    (void)repeatParams;
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddDeqRelu"); });
}

/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 0
namespace RegMulAddDst {
template <typename T, typename U, typename RegT, typename RegU>
__aicore__ inline void MulAddDst(RegT &dstReg, RegU &srcReg0, RegU &srcReg1, Reg::MaskReg &mask)
{
    if constexpr (std::is_same<T, U>::value) {
        Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
    } else {
        Reg::RegTensor<half> fp16RegTemp;
        Reg::RegTensor<float> castReg1, castReg2;
        // the first 64 half is needed to do muladddst in each repeat
        Reg::UnPack<uint32_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(
            (Reg::RegTensor<uint32_t> &)fp16RegTemp, (Reg::RegTensor<uint16_t> &)srcReg0);
        Reg::Cast<float, half, CastParam::mulAddDstTrait>(castReg1, fp16RegTemp, mask);
        Reg::UnPack<uint32_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(
            (Reg::RegTensor<uint32_t> &)fp16RegTemp, (Reg::RegTensor<uint16_t> &)srcReg1);
        Reg::Cast<float, half, CastParam::mulAddDstTrait>(castReg2, fp16RegTemp, mask);
        Reg::MulAddDst(dstReg, castReg1, castReg2, mask);
    }
}
} // namespace RegMulAddDst

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(), "Failed to "
        "check dtype in MulAddDst, current api support dtype combination is src: half, dst: half / float; src: float, "
        "dst: float.");
    Reg::RegTensor<half> fp16RegTemp;
    Reg::RegTensor<float> castReg1, castReg2;
    constexpr auto func = RegMulAddDst::MulAddDst<T, U, Reg::RegTensor<T>, Reg::RegTensor<U>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        mask, 0, repeatTime, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDstImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
    const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(), "Failed to "
        "check dtype in MulAddDst, current api support dtype combination is src: half, dst: half / float; src: float, "
        "dst: float.");
    constexpr auto func = RegMulAddDst::MulAddDst<T, U, Reg::RegTensor<T>, Reg::RegTensor<U>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        nullptr, mask, repeatTime, repeatParams);
}

/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 2
template <typename T, typename U>
__simd_vf__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(),
        "Failed to check dtype in MulAddDst, current api "
        "support dtype combination is src: half, dst: half / float; src: float, dst: float");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(T);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
    Reg::RegTensor<U> src0Reg, src1Reg;
    Reg::RegTensor<T> dstReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = Reg::UpdateMask<T>(sreg);
        Reg::DataCopy(src0Reg, src0 + i * numPerRep);
        Reg::DataCopy(src1Reg, src1 + i * numPerRep);
        Reg::DataCopy(dstReg, dst + i * numPerRep);
        Reg::MulAddDst(dstReg, src0Reg, src1Reg, mask);
        Reg::DataCopy(dst + i * numPerRep, dstReg, mask);
    }
}

__simd_vf__ inline void MulAddDstImpl(__ubuf__ float* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    const int32_t calCount)
{
    uint32_t sregB32 = static_cast<uint32_t>(calCount);     // updated when float calculation
    constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(float);     // each repeat 64 half->float to calculate
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
    Reg::RegTensor<half> src0Reg, src1Reg;
    Reg::RegTensor<float> dstReg, castReg1, castReg2;
    Reg::MaskReg maskB32;                              // updated when float calculation
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskB32 = Reg::UpdateMask<float>(sregB32);
        Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(src0Reg, src0 + i * numPerRep); // 64 half
        Reg::DataCopy<half, Reg::LoadDist::DIST_UNPACK_B16>(src1Reg, src1 + i * numPerRep); // 64 half
        Reg::Cast<float, half, CastParam::mulAddDstTrait>(castReg1, src0Reg, maskB32);           // 64 float
        Reg::Cast<float, half, CastParam::mulAddDstTrait>(castReg2, src1Reg, maskB32);           // 64 float
        Reg::DataCopy(dstReg, dst + i * numPerRep);
        Reg::MulAddDst(dstReg, castReg1, castReg2, maskB32);
        Reg::DataCopy(dst + i * numPerRep, dstReg, maskB32);
    }
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void AddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported AddRelu"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(),
        "Failed to check dtype in FusedMulAdd, current api support dtype "
        "combination is src and dst both: half/float.");
    constexpr auto func = Reg::FusedMulDstAdd<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float>(),
        "Failed to check dtype in FusedMulAdd, current api support dtype "
        "combination is src and dst both: half/float.");
    constexpr auto func = Reg::FusedMulDstAdd<T, Reg::MaskMergeMode::ZEROING, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false, Internal::BinaryFuncMode::DST_SRC_INPUT>(dst, src0, src1,
        nullptr, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__simd_vf__ inline void FusedMulAddImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t calCount)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedMulAdd,"
        "current api support dtype combination is src and dst both: half / float.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vDstReg0;
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vDstReg1;
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg0;
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg1;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
            Reg::DataCopy(vSrcReg0, src0 + i * sregLower);
            Reg::DataCopy(vSrcReg1, src1 + i * sregLower);
            Reg::DataCopy(vDstReg0, dst + i * sregLower);
            Reg::Mul(vDstReg1, vSrcReg0, vDstReg0, mask);
            Reg::Add(vDstReg0, vDstReg1, vSrcReg1, mask);
            Reg::DataCopy(dst + i * sregLower, vDstReg0, mask);
        }
    } else {
        constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        const uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        Reg::RegTensor<T> src0Reg;
        Reg::RegTensor<T> src1Reg;
        Reg::RegTensor<T> dstReg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<T>(sreg);
            Reg::DataCopy(src0Reg, src0 + i * repeatStride);
            Reg::DataCopy(src1Reg, src1 + i * repeatStride);
            Reg::DataCopy(dstReg, dst + i * repeatStride);
            Reg::FusedMulDstAdd(dstReg, src0Reg, src1Reg, mask);
            Reg::DataCopy(dst + i * repeatStride, dstReg, mask);
        }
    }
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask[2], const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t &calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported FusedMulAddRelu"); });
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 0
namespace RegSubRelu {
template <typename T, typename RegT>
__aicore__ inline void SubRelu(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, Reg::MaskReg &mask)
{
    Reg::Sub(dstReg, srcReg0, srcReg1, mask);
    Reg::Maxs(dstReg, dstReg, (T)0, mask);
}
} // namespace RegSubRelu

// SubRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in SubRelu, current api support dtype "
        "combination is src and dst both: half / float / int16_t.");
    constexpr auto func = RegSubRelu::SubRelu<T, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t>(), "Failed to check dtype in SubRelu, current api support dtype "
        "combination is src and dst both: half / float / int16_t.");
    constexpr auto func = RegSubRelu::SubRelu<T, Reg::RegTensor<T>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTimes, repeatParams);
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 2
template <typename T>
__simd_vf__ inline void SubReluImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t calCount)
{
    static_assert(SupportType<T, half, float, int16_t, uint64_t, int64_t>(), "Failed to check dtype in SubRelu, "
        "current api support dtype combination is src and dst both: half / float / int16_t / uint64_t / int64_t.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    const T scalarValue = 0;
    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vDstReg;
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg0;
        Reg::RegTensor<T, Reg::RegTraitNumTwo> vSrcReg1;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
            Reg::DataCopy(vSrcReg0, src0 + i * sregLower);
            Reg::DataCopy(vSrcReg1, src1 + i * sregLower);
            Reg::Sub(vDstReg, vSrcReg0, vSrcReg1, mask);
            Reg::Maxs(vDstReg, vDstReg, scalarValue, mask);
            Reg::DataCopy(dst + i * sregLower, vDstReg, mask);
        }
    } else {
        constexpr uint16_t numPerRep = VECTOR_REG_WIDTH / sizeof(T);
        const uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, numPerRep));
        Reg::RegTensor<T> dstReg, src0Reg, src1Reg;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<T>(sreg);
            Reg::DataCopy(src0Reg, src0 + i * numPerRep);
            Reg::DataCopy(src1Reg, src1 + i * numPerRep);
            Reg::Sub(dstReg, src0Reg, src1Reg, mask);
            Reg::Maxs(dstReg, dstReg, scalarValue, mask);
            Reg::DataCopy(dst + i * numPerRep, dstReg, mask);
        }
    }
}
}

#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_IMPL_H__
#endif
