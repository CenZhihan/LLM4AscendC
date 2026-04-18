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
#include "mockcpp/mockcpp.hpp"

// using namespace std;
using namespace AscendC;

class DataCopyL2CacheTestSuite : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        SetGCoreType(1);
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        SetGCoreType(0);
    }
};

uint8_t GetCacheModeValue(CacheMode mode)
{
    static std::map<CacheMode, uint8_t> map = {
        { CacheMode::CACHE_MODE_DISABLE, 0b100 },
        { CacheMode::CACHE_MODE_NORMAL, 0b000 },
    };

    return map[mode];
}

#define DATA_COPY_L2_CACHE_L1_TEST(T, cacheMode)                                             \
    TEST_F(DataCopyL2CacheTestSuite, testCaseL1##T##cacheMode)                               \
    {                                                                                        \
        uint8_t srcGm[1024]{ 0 };                                                            \
        uint32_t dataSize = 1024 / sizeof(T);                                                \
                                                                                             \
        CacheMode mode = CacheMode::cacheMode;                                               \
        const uint8_t expectedMode = GetCacheModeValue(mode);                                \
        uint8_t actualMode = 99;                                                             \
                                                                                             \
        MOCKER(DataCopyGM2L1Impl, void (*)(T *, T *, const DataCopyParams &, const uint8_t)) \
            .stubs()                                                                         \
            .with(any(), any(), any(), spy(actualMode));                                     \
        MOCKER(DataCopyL12GMImpl, void (*)(T *, T *, const DataCopyParams &, const uint8_t)) \
            .stubs()                                                                         \
            .with(any(), any(), any(), spy(actualMode));                                     \
        TPipe tpipe;                                                                         \
                                                                                             \
        GlobalTensor<T> globalTensor;                                                        \
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);         \
        globalTensor.SetL2CacheHint(mode);                                                   \
                                                                                             \
        TBuf<TPosition::A1> tbuf1;                                                           \
        tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));                                       \
        LocalTensor<T> inputLocal = tbuf1.Get<T>();                                          \
                                                                                             \
        DataCopyParams param;                                                                \
                                                                                             \
        DataCopy(inputLocal, globalTensor, param);                                           \
        EXPECT_EQ(expectedMode, actualMode);                                                 \
                                                                                             \
        actualMode = 99;                                                                     \
        DataCopy(globalTensor, inputLocal, param);                                           \
        EXPECT_EQ(expectedMode, actualMode);                                                 \
    }

DATA_COPY_L2_CACHE_L1_TEST(uint16_t, CACHE_MODE_NORMAL)
DATA_COPY_L2_CACHE_L1_TEST(int8_t, CACHE_MODE_DISABLE)

#define LOAD_DATA_L2_CACHE_TEST(T, cacheMode, pos)                                                \
    TEST_F(DataCopyL2CacheTestSuite, testCaseL1##T##cacheMode##pos)                               \
    {                                                                                             \
        uint8_t srcGm[1024]{ 0 };                                                                 \
        uint32_t dataSize = 1024 / sizeof(T);                                                     \
                                                                                                  \
        CacheMode mode = CacheMode::cacheMode;                                                    \
        const uint8_t expectedMode = GetCacheModeValue(mode);                                     \
        uint8_t actualMode = 99;                                                                  \
                                                                                                  \
        MOCKER(LoadData2DGM2L1Cal, void (*)(T *, T *, const LoadData2DParamsV2 &, const uint8_t)) \
            .stubs()                                                                              \
            .with(any(), any(), any(), spy(actualMode));                                          \
        TPipe tpipe;                                                                              \
                                                                                                  \
        GlobalTensor<T> globalTensor;                                                             \
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);              \
        globalTensor.SetL2CacheHint(mode);                                                        \
                                                                                                  \
        TBuf<TPosition::pos> tbuf1;                                                               \
        tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));                                            \
        LocalTensor<T> inputLocal = tbuf1.Get<T>();                                               \
                                                                                                  \
        LoadData2DParamsV2 param;                                                                 \
                                                                                                  \
        LoadData(inputLocal, globalTensor, param);                                                \
        EXPECT_EQ(expectedMode, actualMode);                                                      \
    }
