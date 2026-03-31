/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_mrgsort4_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/c_api/instr_impl/npu_arch_3510/vector_compute_impl/asc_mrgsort4_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "c_api/asc_simd.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_MRGSORT4_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_MRGSORT4_IMPL_H

#include "instr_impl/npu_arch_3510/utils_impl/utils_impl.h"

__aicore__ inline void asc_mrgsort4_impl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat,
                                         uint16_t region_proposal_li0, uint16_t region_proposal_li1,
                                         uint16_t region_proposal_li2, uint16_t region_proposal_li3,
                                         bool is_all_stored, uint8_t mask_signal)
{
    if ASC_IS_AIV {
        vmrgsort4(dst, src, repeat, region_proposal_li0, region_proposal_li1, region_proposal_li2, region_proposal_li3,
                  is_all_stored, mask_signal);
    }
}

__aicore__ inline void asc_mrgsort4_sync_impl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat,
                                              uint16_t region_proposal_li0, uint16_t region_proposal_li1,
                                              uint16_t region_proposal_li2, uint16_t region_proposal_li3,
                                              bool is_all_stored, uint8_t mask_signal)
{
    asc_mrgsort4_impl(dst, src, repeat, region_proposal_li0, region_proposal_li1, region_proposal_li2, region_proposal_li3,
                      is_all_stored, mask_signal);
    asc_sync_post_process();
}

__aicore__ inline void asc_mrgsort4_impl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat,
                                         uint16_t region_proposal_li0, uint16_t region_proposal_li1,
                                         uint16_t region_proposal_li2, uint16_t region_proposal_li3,
                                         bool is_all_stored, uint8_t mask_signal)
{
    if ASC_IS_AIV {
        vmrgsort4(dst, src, repeat, region_proposal_li0, region_proposal_li1, region_proposal_li2, region_proposal_li3,
                  is_all_stored, mask_signal);
    }
}

__aicore__ inline void asc_mrgsort4_sync_impl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat,
                                              uint16_t region_proposal_li0, uint16_t region_proposal_li1,
                                              uint16_t region_proposal_li2, uint16_t region_proposal_li3,
                                              bool is_all_stored, uint8_t mask_signal)
{
    asc_mrgsort4_impl(dst, src, repeat, region_proposal_li0, region_proposal_li1, region_proposal_li2, region_proposal_li3,
                      is_all_stored, mask_signal);
    asc_sync_post_process();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
