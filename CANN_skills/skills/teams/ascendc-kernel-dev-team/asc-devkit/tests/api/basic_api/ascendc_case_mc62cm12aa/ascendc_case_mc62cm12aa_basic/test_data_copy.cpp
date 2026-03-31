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
#define private public
#define protect public
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

struct TestDataCopyParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t, bool);
    bool ub2l1;
};

class TestDataCopySuite : public testing::Test, public testing::WithParamInterface<TestDataCopyParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template<typename T>
void main_data_copy_kernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, int32_t dataSize, bool ub2l1)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    LocalTensor<T> inputLocal;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECIN;
    inputLocal.SetAddr(tbuf);
    inputLocal.InitBuffer(0, dataSize);
    LocalTensor<T> outputLocal;
    TBuffAddr tbuf_out;
    tbuf_out.logicPos = (uint8_t)TPosition::VECOUT;
    outputLocal.SetAddr(tbuf_out);
    outputLocal.InitBuffer(0, dataSize);
    LocalTensor<T> l1Local;
    TBuffAddr tbuf_l1;
    tbuf_l1.logicPos = (uint8_t)TPosition::A1;
    l1Local.SetAddr(tbuf_l1);
    l1Local.InitBuffer(0, dataSize);


    LocalTensor<T> biasLocal;
    TBuffAddr tbuf_bias;
    tbuf_bias.logicPos = (uint8_t)TPosition::C2;  
    biasLocal.SetAddr(tbuf_bias);
    biasLocal.InitBuffer(0, dataSize);

    DataCopyParams datacopyParams{1, 32 * sizeof(T), 0, 0};
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(inputLocal, srcGlobal, datacopyParams);
    if (ub2l1) {
        DataCopy(l1Local, inputLocal, datacopyParams, enhancedParams);
    } else {
        DataCopy(outputLocal, inputLocal, datacopyParams, enhancedParams);
    }
    DataCopy(dstGlobal,inputLocal,datacopyParams);
    DataCopy(biasLocal, l1Local, datacopyParams);
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY, TestDataCopySuite,
    ::testing::Values(TestDataCopyParams { 512, 4, main_data_copy_kernel<int32_t>, false },
    TestDataCopyParams { 512, 2, main_data_copy_kernel<int16_t>, false },
    TestDataCopyParams { 512, 4, main_data_copy_kernel<float>, false },
    TestDataCopyParams { 768, 2, main_data_copy_kernel<half>, false },
    TestDataCopyParams { 2048, 1, main_data_copy_kernel<int8_t>, false },
    TestDataCopyParams { 2048, 1, main_data_copy_kernel<uint8_t>, false },
    TestDataCopyParams { 512, 4, main_data_copy_kernel<int32_t>, true },
    TestDataCopyParams { 512, 2, main_data_copy_kernel<int16_t>, true },
    TestDataCopyParams { 512, 4, main_data_copy_kernel<float>, true },
    TestDataCopyParams { 768, 2, main_data_copy_kernel<half>, true },
    TestDataCopyParams { 2048, 1, main_data_copy_kernel<int8_t>, true },
    TestDataCopyParams { 2048, 1, main_data_copy_kernel<uint8_t>, true }));

TEST_P(TestDataCopySuite, TestDataCopyCases)
{
    auto param = GetParam();
    uint8_t* srcGm = new uint8_t[param.dataSize * param.typeSize * 10];
    uint8_t* dstGm = new uint8_t[param.dataSize * param.typeSize * 10];
    param.cal_func(srcGm, dstGm, param.dataSize, param.ub2l1);
    delete [] srcGm;
    delete [] dstGm;
}

class DataCopyUB2L0CTestSuite : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        SetGCoreType(2);
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        SetGCoreType(0);
    }
};
 
TEST_F(DataCopyUB2L0CTestSuite, testCaseDataCopyUB2L0C)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    DataCopyEnhancedParams enhancedParams;
    MOCKER(raise).stubs().will(returnValue(int(0)));
    DataCopyUB2L0CIntf(dstGm, srcGm, params, enhancedParams);
}