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
using AscendC::Reg::RegTensor;
using AscendC::Reg::MaskReg;
using AscendC::Reg::UpdateMask;

template <typename T, uint8_t Mode>
class KernelGmAccess {
public:
    __aicore__ inline KernelGmAccess()
    {}

    __aicore__ inline void Process(GM_ADDR dst0_gm, GM_ADDR src0_gm, uint32_t calcount)
    {
        if (Mode == 0) {
            T value = ReadGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(src0_gm));
            WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(dst0_gm), value);
        } else {
            T value = ReadGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(src0_gm + sizeof(T)));
            WriteGmByPassDCache<T>(reinterpret_cast<__gm__ T *>(dst0_gm), value);
        }
    }
 
private:
    GlobalTensor<T> src0_global;
    GlobalTensor<T> dst0_global;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    uint32_t dataSize = 0;
};
 
struct MicroGmAccessParams {
    void (*CallFunc)();
};
 
template<typename T, uint8_t Mode>
void RunCase() {
    int src_byte_size = sizeof(T);
    int dst_byte_size = sizeof(T);
    int data_size = 256;
    uint8_t dstGm0[data_size * dst_byte_size] = {0};
    uint8_t srcGm0[data_size * src_byte_size] = {0};
 
    KernelGmAccess<T, Mode> op;
    op.Process(dstGm0, srcGm0, data_size);
}

class MicroGmAccessTestSuite : public testing::Test, public testing::WithParamInterface<MicroGmAccessParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};
 
INSTANTIATE_TEST_CASE_P(MicroGmAccessTestCases, MicroGmAccessTestSuite,
    ::testing::Values(MicroGmAccessParams { RunCase<uint64_t, 0>},
                      MicroGmAccessParams { RunCase<int64_t, 0>},
                      MicroGmAccessParams { RunCase<uint32_t, 0>},
                      MicroGmAccessParams { RunCase<int32_t, 0>},
                      MicroGmAccessParams { RunCase<uint16_t, 0>},
                      MicroGmAccessParams { RunCase<int16_t, 0>},
                      MicroGmAccessParams { RunCase<uint8_t, 0>},
                      MicroGmAccessParams { RunCase<int8_t, 0>},
                      MicroGmAccessParams { RunCase<uint64_t, 1>},
                      MicroGmAccessParams { RunCase<int64_t, 1>},
                      MicroGmAccessParams { RunCase<uint32_t, 1>},
                      MicroGmAccessParams { RunCase<int32_t, 1>},
                      MicroGmAccessParams { RunCase<uint16_t, 1>},
                      MicroGmAccessParams { RunCase<int16_t, 1>},
                      MicroGmAccessParams { RunCase<uint8_t, 1>},
                      MicroGmAccessParams { RunCase<int8_t, 1>}
                      ));
 
TEST_P(MicroGmAccessTestSuite, MicroGmAccessTestCase)
{
    auto param = GetParam();
    param.CallFunc();
}