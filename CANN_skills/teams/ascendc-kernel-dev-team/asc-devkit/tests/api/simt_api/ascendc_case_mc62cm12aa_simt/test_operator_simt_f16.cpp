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
#include <type_traits>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128
template <typename T>
class KernelFP16 {
    public:
        __aicore__ KernelFP16() {}
        __aicore__ inline void Process(__gm__ T* dst, int32_t mode);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelFP16Compute(__gm__ T* dst, int32_t mode)
{
    for(int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>();idx < 1; idx+=block_num*GetThreadNum<0>()) {
        if(mode == 0) {
            half a = 2.3;
            a++;
            dst[idx] = ++a;
        } else if(mode == 1) {
            half a = 2.1;
            a--;
            dst[idx] = --a;
        } else if(mode == 2) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a && b;
        } else if(mode == 3) {
            half a = 2.1;
            half b = 0.0;
            dst[idx] = a || b;
        } else if(mode == 4) {
            half a = 2.1;
            dst[idx] = Floor(a);
        } else if(mode == 5) {
            half a = 2.1;
            dst[idx] = Rint(a);
        } else if(mode == 6) {
            half a = 2.1;
            dst[idx] = Ceil(a);
        }
    }
}

template <typename T>
__aicore__ inline void KernelFP16<T>::Process(__gm__ T* dst, int32_t mode)
{
    AscendC::Simt::VF_CALL<KernelFP16Compute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, mode);
}



struct FP16Params {
    int32_t mode;
};

class FP16Testsuite : public testing::Test, public testing::WithParamInterface<FP16Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};


INSTANTIATE_TEST_CASE_P(FP16TestCase, FP16Testsuite,
    ::testing::Values(FP16Params { 0 }, FP16Params { 1 }, FP16Params { 2 }, FP16Params { 3 },
            FP16Params { 4 }, FP16Params { 5 }, FP16Params { 6 } ));

TEST_P(FP16Testsuite, FP16TestCase)
{
    auto param = GetParam();
    int32_t mode = param.mode;
    int fp_byte_size = 2;
    int shape_size = 128;

    uint8_t dstGm[shape_size * fp_byte_size] = {0};
    KernelFP16<uint8_t> op;
    op.Process((__gm__ uint8_t*)dstGm, mode);
}
