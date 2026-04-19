/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
using namespace std;

namespace AscendC {
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
    PolicyType POLICY_TYPE = PolicyType::MATMUL_DEFAULT, typename = void>
class CustomMatmulScheduler
{
    using DstT = typename C_TYPE::T;
public:
    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe) {}
    __aicore__ inline bool ScheduleOnce(bool enPartialSum) { return true; }
    __aicore__ inline void GetResult(const GlobalTensor<DstT> &gm, uint8_t enAtomic = 0,
                                     bool enSequentialWrite = false) {}
    __aicore__ inline void End() {}
    __aicore__ inline void Reset() {}
};
 
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
class CustomBatchScheduler
{
public:
    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe) {}
    template<class T>
    void Schedule(const T& dst, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite,
        const uint32_t matrixStrideA, const uint32_t matrixStrideB, const uint32_t matrixStrideC) {};
    __aicore__ inline void End() {}
};
}
