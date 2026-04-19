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

using namespace std;
using namespace AscendC;

class TEST_INIT_GLOBAL_MEMORY : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

template <typename T>
void main_init_global_memory_demo(__gm__ uint8_t *__restrict__ dst_gm, const uint64_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> dst_global;
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm), dataSize);
    InitGlobalMemory(dst_global, dataSize, (T)10);
    PipeBarrier<PIPE_ALL>();
}
#define VEC_INIT_GLOBAL_MEMORY_TESTCASE(DATASIZE, DATA_TYPE)                                    \
    TEST_F(TEST_INIT_GLOBAL_MEMORY, INIT_GLOBAL_MEMORY##_##DATASIZE##_##DATA_TYPE##_##Case)     \
    {                                                                                           \
        uint8_t output_gm[DATASIZE * sizeof(DATA_TYPE)] = {0};                                  \
        main_init_global_memory_demo<DATA_TYPE>(output_gm, DATASIZE);                           \
        for (uint32_t i = 0; i < DATASIZE; i++) {                                               \
                EXPECT_EQ(output_gm[i], 0x00);                                                  \
        }                                                                                       \
    }

VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, half);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, half);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, float);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, float);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, uint16_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, uint16_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, int16_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, int16_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, uint32_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, uint32_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8192, int32_t);
VEC_INIT_GLOBAL_MEMORY_TESTCASE(8193, int32_t);