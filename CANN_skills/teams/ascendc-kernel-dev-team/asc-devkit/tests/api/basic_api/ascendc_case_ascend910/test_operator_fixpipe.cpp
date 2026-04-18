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
#include "test_utils.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;

namespace AscendC {

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
int32_t RaiseStubForFixpipe(int32_t input)
{
    return 0;
}

#define KERNEL_FIXPIPE_FAILED(l1outT, dstT, name)                                     \
    extern "C" __global__ __aicore__ void kernel_fixpipe_##name(bool isL0C2OUt)         \
    {                                                                                   \
        GlobalTensor<dstT> outputGm;                                                  \
        LocalTensor<l1outT> l0CLocal;                                                  \
        FixpipeParams<l1outT> fixpipeParams;                                           \
        LocalTensor<dstT> l1Local;                                                     \
        MOCKER(raise, int32_t (*)(int32_t)).stubs().will(invoke(RaiseStubForFixpipe));  \
        if (isL0C2OUt) {                                                                \
            Fixpipe(outputGm, l0CLocal, fixpipeParams);                                \
        } else {                                                                        \
            Fixpipe(l1Local, l0CLocal, fixpipeParams);                                  \
        }                                                                               \
        LocalTensor<dstT> reluPre;                                                     \
        LocalTensor<dstT> TensorPre;                                                   \
        LocalTensor<dstT> quantPre;                                                    \
        bool isUnitFlag = false;                                                        \
        SetFixPipeConfig(reluPre, quantPre, isUnitFlag);                                \
        SetFixPipeConfig(TensorPre, isUnitFlag);                                        \
        SetFixpipeNz2ndFlag(0, 0, 0);                                                   \
        SetFixpipePreQuantFlag(0);                                                      \
    }  // namespace AscendC

KERNEL_FIXPIPE_FAILED(int32_t, half, check_func_failed)
struct FixpipeTestParams {
    void (*cal_func)(bool);
    bool isL0C2OUt;
};

class FixpipeTestsuite : public testing::Test, public testing::WithParamInterface<FixpipeTestParams> {
protected:
    void SetUp() {}
    void TearDown() {
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_FIXPIPE, FixpipeTestsuite,
    ::testing::Values(
    FixpipeTestParams { kernel_fixpipe_check_func_failed, true },
    FixpipeTestParams { kernel_fixpipe_check_func_failed, false }));

TEST_P(FixpipeTestsuite, FixpipeTestCase)
{
    auto param = GetParam();
    param.cal_func(param.isL0C2OUt);
}
} // namespace AscendC