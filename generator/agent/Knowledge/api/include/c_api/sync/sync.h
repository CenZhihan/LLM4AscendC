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
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif     

#ifndef INCLUDE_C_API_SYNC_SYNC_H
#define INCLUDE_C_API_SYNC_SYNC_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)

#include "instr_impl/npu_arch_2201/sync_impl.h"

#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

#include "instr_impl/npu_arch_3510/sync_impl.h"

#endif

#define asc_sync_notify(pipe, tpipe, id) asc_sync_notify_impl(pipe, tpipe, id)

#define asc_sync_wait(pipe, tpipe, id) asc_sync_wait_impl(pipe, tpipe, id)

#define asc_sync_pipe(pipe) asc_sync_pipe_impl(pipe)

__aicore__ inline void asc_sync_vec();

__aicore__ inline void asc_sync_mte3(int id);

__aicore__ inline void asc_sync_mte2(int id);

__aicore__ inline void asc_sync();

#define asc_sync_block_arrive(pipe, flag_id) asc_sync_block_arrive_impl((pipe), (flag_id))

#define asc_sync_data_barrier(arg) asc_sync_data_barrier_impl((arg))

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)

#define asc_sync_block_wait(pipe, flagID) wait_flag_dev((flagID))

#define asc_sync_inter_wait(pipe, flagID) wait_flag_dev((flagID))

#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

#define asc_sync_intra_arrive(pipe, sync_id) set_intra_block((pipe), (sync_id))

#define asc_sync_intra_wait(pipe, sync_id) wait_intra_block((pipe), (sync_id))

#define asc_sync_inter_wait(pipe, flag_id) wait_flag_dev((pipe), (flag_id))

#define asc_sync_block_wait(pipe, flag_id) wait_flag_dev((pipe), (flag_id))

#define asc_release_buf(pipe, buf_id, mode) rls_buf((pipe), (buf_id), (mode))

#define asc_get_buf(pipe, buf_id, mode) asc_get_buf_impl((pipe), (buf_id), (mode))

#endif

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)  
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS  
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H  
#endif    
