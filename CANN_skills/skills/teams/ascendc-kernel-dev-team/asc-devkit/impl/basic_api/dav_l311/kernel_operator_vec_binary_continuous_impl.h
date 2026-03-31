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
 * \file kernel_operator_vec_binary_continuous_impl.h
 * \brief AscendC l311 support vec binary continuous data api.
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#pragma message("impl/basic_api/dav_l311/kernel_operator_vec_binary_continuous_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_vec_intf.h\"\" and use public functions or variables defined in interface headers files.")
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H__
#endif
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {

// for Level 2 binary op
#define BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(FUNC_NAME)                                                           \
    template <typename T>                                                                                          \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count) \
    {                                                                                                              \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });               \
    }

// for Level 2 binary op
#define BINARY_OP_CONTINUOUS_IMPL(FUNC_NAME, OP_NAME, DataType)                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ DataType* dst, __ubuf__ DataType* src0, __ubuf__ DataType* src1,     \
                                     const int32_t& count)                                                      \
    {                                                                                                              \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            RegTensor<DataType> vreg0;                                                                             \
            RegTensor<DataType> vreg1;                                                                             \
            RegTensor<DataType> vreg2;                                                                             \
            uint32_t sreg = (uint32_t)count;                                                                    \
            MaskReg preg;                                                                                          \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(DataType));                                  \
            uint16_t repeatTime = CeilDivision(count, sregLower);                                              \
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {                                                 \
                preg = CreatePredicate<DataType>(sreg);                                                            \
                DataCopy(vreg0, src0, i * sregLower);                                                              \
                DataCopy(vreg1, src1, i * sregLower);                                                              \
                OP_NAME(vreg2, vreg0, vreg1, preg);                                                                \
                DataCopy(dst, vreg2,  i * sregLower, preg);                                                         \
            }                                                                                                      \
        }                                                                                                          \
    }

/* **************************************************************************************************
 * Add                                                                                              *
 * **************************************************************************************************/
// Add::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(AddImpl)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, int8_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, int16_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, int32_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, half)
BINARY_OP_CONTINUOUS_IMPL(AddImpl, Add, float)

/* **************************************************************************************************
 * Sub                                                                                              *
 * **************************************************************************************************/
// Sub::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(SubImpl)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, int8_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, int16_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, int32_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, half)
BINARY_OP_CONTINUOUS_IMPL(SubImpl, Sub, float)

/* **************************************************************************************************
 * Mul                                                                                              *
 * **************************************************************************************************/
// Mul::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(MulImpl)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, int8_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, int16_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, int32_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, half)
BINARY_OP_CONTINUOUS_IMPL(MulImpl, Mul, float)
/* **************************************************************************************************
 * Div                                                                                              *
 * **************************************************************************************************/
// Div::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(DivImpl)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, int16_t)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, int32_t)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, half)
BINARY_OP_CONTINUOUS_IMPL(DivImpl, Div, float)

/* **************************************************************************************************
 * Max                                                                                              *
 * **************************************************************************************************/
// Max::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(MaxImpl)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, int16_t)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, int32_t)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, half)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, float)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, int8_t)
BINARY_OP_CONTINUOUS_IMPL(MaxImpl, Max, uint8_t)
/* **************************************************************************************************
 * Min                                                                                              *
 * **************************************************************************************************/
// Min::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(MinImpl)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, int8_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, int16_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, int32_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, uint32_t)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, half)
BINARY_OP_CONTINUOUS_IMPL(MinImpl, Min, float)
/* **************************************************************************************************
 * And                                                                                              *
 * **************************************************************************************************/
// And::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(AndImpl)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, int8_t)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, int16_t)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, int32_t)
BINARY_OP_CONTINUOUS_IMPL(AndImpl, And, uint32_t)
/* **************************************************************************************************
 * Or                                                                                               *
 * **************************************************************************************************/
// Or::Level 2
BINARY_OP_CONTINUOUS_IMPL_NOT_SUPPORT(OrImpl)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, int8_t)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, uint8_t)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, int16_t)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, uint16_t)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, int32_t)
BINARY_OP_CONTINUOUS_IMPL(OrImpl, Or, uint32_t)
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H__
#endif
