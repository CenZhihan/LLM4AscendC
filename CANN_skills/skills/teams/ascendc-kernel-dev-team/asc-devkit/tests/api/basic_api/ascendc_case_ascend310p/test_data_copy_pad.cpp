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
// #include "api_check/kernel_cpu_check.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

class DataCopyUB2L1TestSuite : public ::testing::Test {
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
 
TEST_F(DataCopyUB2L1TestSuite, testCaseDataCopyUB2L1)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    DataCopyL12UBImpl((__cbuf__ uint16_t*)dstGm.GetPhyAddr(), (__ubuf__ uint16_t*)srcGm.GetPhyAddr(), params);
}

class DataCopyL12BTTestSuite : public ::testing::Test {
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
 
TEST_F(DataCopyL12BTTestSuite, testCaseDataCopyL12BT)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    DataCopyL12BTImpl((uint64_t)dstGm.GetPhyAddr(), (__cbuf__ uint16_t*)srcGm.GetPhyAddr(), static_cast<uint16_t>(0), params);
}