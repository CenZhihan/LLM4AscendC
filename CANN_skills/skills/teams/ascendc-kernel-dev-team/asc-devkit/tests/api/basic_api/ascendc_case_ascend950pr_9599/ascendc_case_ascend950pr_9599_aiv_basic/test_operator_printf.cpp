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
#include <mockcpp/mockcpp.hpp>
#include "kernel_operator.h"

using namespace AscendC;
#define ASCENDC_DUMP

enum class PrintfCaseEnum : uint32_t {
    VALUE = 9
};

class TestPrintfSuite : public testing::Test {
protected:
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

int32_t RaiseStubForPrintf(int32_t input)
{
    return 0;
}

void DataPrintfCase(__gm__ uint8_t* srcGm, __gm__ uint8_t* workGm, __gm__ uint32_t dataSize, __gm__ uint64_t dumpSize)
{
    GlobalTensor<uint8_t> srcGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcGm), dataSize);

    InitDump(workGm, dumpSize);
    PrintTimeStamp(0);

    const int32_t signedValue = -3;
    const uint32_t unsignedValue = 6U;
    const float floatValue = 1.25F;
    const double doubleValue = 2.5;
    const PrintfCaseEnum enumValue = PrintfCaseEnum::VALUE;
    __gm__ uint8_t* ptrValue = srcGm;

    PRINTF("PRINTF_CASE %s %d %u", "hello_printf", signedValue, unsignedValue);
    PRINTF("PRINTF_CASE2 %f %lf %p %d", floatValue, doubleValue, static_cast<void*>(ptrValue),
        static_cast<int32_t>(enumValue));
}

TEST_F(TestPrintfSuite, PrintfCase)
{
    int32_t tmp = g_coreType;
    g_coreType = 2;

    constexpr uint32_t dataSize = 64;
    constexpr uint64_t dumpSize = 96 * 3;
    uint8_t srcGm[dataSize] = {0};
    uint8_t workGm[dumpSize * sizeof(uint32_t)] = {0};
    MOCKER(raise, int32_t (*)(int32_t)).stubs().will(invoke(RaiseStubForPrintf));

    DataPrintfCase(srcGm, workGm, dataSize, dumpSize);

    g_coreType = tmp;
}
