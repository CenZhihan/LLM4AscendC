/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)  
#warning "impl/c_api/c_api_instr_impl/utils_c_api_impl/utils_c_api_type.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."  
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC  
#endif    

#ifndef IMPL_C_API_C_API_INSTR_IMPL_UTILS_C_API_IMPL_UTILS_C_API_TYPE_H
#define IMPL_C_API_C_API_INSTR_IMPL_UTILS_C_API_IMPL_UTILS_C_API_TYPE_H

#include "impl/utils/sys_macros.h"

constexpr uint32_t INT4_BIT_NUM = 4;

template <bool condition, class T1, class T2>
struct Conditional {
    using type = T1;
};

template <int bit_num, bool sign = true>
struct integer_sub_type {
    static int const k_bits = bit_num;
    static bool const k_signed = sign;

    using T = typename Conditional<k_signed, int8_t, uint8_t>::type;
    using Storage = uint8_t;

    static Storage const mask = Storage(((static_cast<uint64_t>(1)) << static_cast<uint32_t>(k_bits)) - 1);
    Storage storage;
    __aicore__ inline integer_sub_type() = default;

    __aicore__ inline integer_sub_type(uint32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline integer_sub_type(int32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline operator T() const
    {
        if (k_signed && ((storage & Storage(static_cast<uint64_t>(1) << static_cast<uint32_t>(k_bits - 1))) != 0)) {
            // Sign extend
            return T(storage) | ~T(mask);
        }
        return T(storage);
    }

    __aicore__ inline bool operator == (integer_sub_type const &rhs) const
    {
        return storage == rhs.storage;
    }

    __aicore__ inline bool operator != (integer_sub_type const &rhs) const
    {
        return storage != rhs.storage;
    }

    __aicore__ inline bool operator > (integer_sub_type const &rhs) const
    {
        bool lhs_is_neg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->k_bits - 1)));
        bool rhs_is_neg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.k_bits - 1)));
        if (k_signed && (lhs_is_neg != rhs_is_neg)) {
            return (!lhs_is_neg) && rhs_is_neg;
        }
        return this->storage > rhs.storage;
    }

    __aicore__ inline bool operator >= (integer_sub_type const &rhs) const
    {
        bool lhs_is_neg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->k_bits - 1)));
        bool rhs_is_neg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.k_bits - 1)));
        if (k_signed && (lhs_is_neg != rhs_is_neg)) {
            return (!lhs_is_neg) && rhs_is_neg;
        }
        return storage >= rhs.storage;
    }

    __aicore__ inline bool operator < (integer_sub_type const &rhs) const
    {
        return !(*this >= rhs);
    }

    __aicore__ inline bool operator <= (integer_sub_type const &rhs) const
    {
        return !(*this > rhs);
    }
};

using int4b_t = integer_sub_type<INT4_BIT_NUM, true>;

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)  
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC  
#endif  

