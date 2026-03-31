/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_simt_atomic_impl.h
 * \brief
 */
#ifndef IMPL_SIMT_API_CPP_DAV_C310_KERNEL_SIMT_ATOMIC_IMPL_H
#define IMPL_SIMT_API_CPP_DAV_C310_KERNEL_SIMT_ATOMIC_IMPL_H

#if defined(ASCENDC_CPU_DEBUG)
#include "kernel_process_lock.h"
#include "kernel_utils.h"
#include "kernel_simt_cpu.h"
#include "stub_def.h"
#endif

namespace AscendC {
namespace Simt {
#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicCasImpl(__gm__ T *address, T compare, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, compare, val]() {
        if (*address == compare) {
            *address = val;
        }
    });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicCasImpl(__ubuf__ T *address, T compare, T val)
{
    return bisheng::cce::simt::atomicCAS(address, compare, val);
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicCasImpl(__gm__ T *address, T compare, T val)
{
    return bisheng::cce::simt::atomicCAS(address, compare, val);
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicAddImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() { *address += val; });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicAddImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<float>>()) {
        return bisheng::cce::simt::atomicAdd(address, val);
    } else {
        bisheng::cce::simt::atomicAdd(address, val);
        return *address;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicAddImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<int64_t>, Tuple<uint64_t>, Tuple<float>,
                              Tuple<half2>, Tuple<bfloat16x2_t>>()) {
        return bisheng::cce::simt::atomicAdd(address, val);
    } else {
        bisheng::cce::simt::atomicAdd(address, val);
        return *address;
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicSubImpl(__gm__ T *address, T val)
{
    return AtomicAddImpl(address, (T)0 - val);
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicSubImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<float>>()) {
        return bisheng::cce::simt::atomicSub(address, val);
    } else {
        bisheng::cce::simt::atomicAdd(address, -val);
        return *address;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicSubImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<int64_t>, Tuple<uint64_t>, Tuple<float>,
                              Tuple<half2>, Tuple<bfloat16x2_t>>()) {
        return bisheng::cce::simt::atomicSub(address, val);
    } else {
        bisheng::cce::simt::atomicAdd(address, -val);
        return *address;
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicExchImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() { *address = val; });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicExchImpl(__ubuf__ T *address, T val)
{
    return bisheng::cce::simt::atomicExch(address, val);
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicExchImpl(__gm__ T *address, T val)
{
    return bisheng::cce::simt::atomicExch(address, val);
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicMaxImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() {
        if (*address < val) {
            *address = val;
        }
    });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicMaxImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<float>>()) {
        return bisheng::cce::simt::atomicMax(address, val);
    } else {
        bisheng::cce::simt::atomicMax(address, val);
        return *address;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicMaxImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<int64_t>, Tuple<uint64_t>, Tuple<float>,
                              Tuple<half2>, Tuple<bfloat16x2_t>>()) {
        return bisheng::cce::simt::atomicMax(address, val);
    } else {
        bisheng::cce::simt::atomicMax(address, val);
        return *address;
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicMinImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() {
        if (*address > val) {
            *address = val;
        }
    });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__simt_callee__ __aicore__ inline T AtomicMinImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<float>>()) {
        return bisheng::cce::simt::atomicMin(address, val);
    } else {
        bisheng::cce::simt::atomicMin(address, val);
        return *address;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T AtomicMinImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<int32_t>, Tuple<uint32_t>, Tuple<int64_t>, Tuple<uint64_t>, Tuple<float>,
                              Tuple<half2>, Tuple<bfloat16x2_t>>()) {
        return bisheng::cce::simt::atomicMin(address, val);
    } else {
        bisheng::cce::simt::atomicMin(address, val);
        return *address;
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicIncImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() {
        if (*address >= val) {
            *address = (T)0;
        } else {
            *address += (T)1;
        }
    });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename DstType, typename SrcType>
__aicore__ inline DstType AtomicIncImpl_(SrcType *address, DstType val)
{
    DstType old = *address;
    DstType cmp;
    DstType newVal;
    do {
        cmp = old;
        if (old >= val) {
            newVal = (DstType)0;
        } else {
            newVal = old + (DstType)1;
        }
        old = AtomicCasImpl(address, cmp, newVal);
    } while (cmp != old);
    return old;
}

template <typename T>
__aicore__ inline T AtomicIncImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<uint32_t>>()) {
        return bisheng::cce::simt::atomicInc(address, val);
    } else {
        return AtomicIncImpl_<T, __ubuf__ T>(address, val);
    }
}

template <typename T>
__aicore__ inline T AtomicIncImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<uint32_t>, Tuple<uint64_t>>()) {
        return bisheng::cce::simt::atomicInc(address, val);
    } else {
        return AtomicIncImpl_<T, __gm__ T>(address, val);
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicDecImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val]() {
        if (*address == (T)0 || *address > val) {
            *address = val;
        } else {
            *address -= (T)1;
        }
    });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename DstType, typename SrcType>
__aicore__ inline DstType AtomicDecImpl_(SrcType *address, DstType val)
{
    DstType old = *address;
    DstType cmp;
    DstType newVal;
    do {
        cmp = old;
        if (old == (DstType)0 || old > val) {
            newVal = val;
        } else {
            newVal = old - (DstType)1;
        }
        old = AtomicCasImpl(address, cmp, newVal);
    } while (cmp != old);
    return old;
}

template <typename T>
__aicore__ inline T AtomicDecImpl(__ubuf__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<uint32_t>>()) {
        return bisheng::cce::simt::atomicDec(address, val);
    } else {
        return AtomicDecImpl_<T, __ubuf__ T>(address, val);
    }
}

template <typename T>
__aicore__ inline T AtomicDecImpl(__gm__ T *address, T val)
{
    if constexpr (SupportType<Tuple<T>, Tuple<uint32_t>, Tuple<uint64_t>>()) {
        return bisheng::cce::simt::atomicDec(address, val);
    } else {
        return AtomicDecImpl_<T, __gm__ T>(address, val);
    }
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicAndImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val, ret]() { *address = ret & val; });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__aicore__ inline T AtomicAndImpl(__ubuf__ T *address, T val)
{
    return bisheng::cce::simt::atomicAnd(address, val);
}

template <typename T>
__aicore__ inline T AtomicAndImpl(__gm__ T *address, T val)
{
    return bisheng::cce::simt::atomicAnd(address, val);
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicOrImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val, ret]() { *address = ret | val; });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__aicore__ inline T AtomicOrImpl(__ubuf__ T *address, T val)
{
    return bisheng::cce::simt::atomicOr(address, val);
}

template <typename T>
__aicore__ inline T AtomicOrImpl(__gm__ T *address, T val)
{
    return bisheng::cce::simt::atomicOr(address, val);
}
#endif

#if defined(ASCENDC_CPU_DEBUG)
template <typename T>
T AtomicXorImpl(__gm__ T *address, T val)
{
    T ret;
    ProcessLock::GetProcessLock()->Write();
    ret = *address;
    ThreadBlock::GetBlockInstance().AtomicOp([address, val, ret]() { *address = ret ^ val; });
    ProcessLock::GetProcessLock()->Unlock();
    return ret;
}
#else
template <typename T>
__aicore__ inline T AtomicXorImpl(__ubuf__ T *address, T val)
{
    return bisheng::cce::simt::atomicXOr(address, val);
}

template <typename T>
__aicore__ inline T AtomicXorImpl(__gm__ T *address, T val)
{
    return bisheng::cce::simt::atomicXOr(address, val);
}
#endif

}  // namespace Simt
}  // namespace AscendC
#endif  // IMPL_SIMT_API_CPP_DAV_C310_KERNEL_SIMT_ATOMIC_IMPL_H
