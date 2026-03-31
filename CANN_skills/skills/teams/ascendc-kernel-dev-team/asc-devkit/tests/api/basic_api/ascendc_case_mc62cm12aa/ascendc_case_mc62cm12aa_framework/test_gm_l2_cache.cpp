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
        SetGCoreType(2);
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

#define DATA_COPY_L2_CACHE_UB_TEST(T, cacheMode)                                             \
    TEST_F(DataCopyL2CacheTestSuite, testCaseUB##T##cacheMode)                               \
    {                                                                                        \
        uint8_t srcGm[1024]{ 0 };                                                            \
        uint32_t dataSize = 1024 / sizeof(T);                                                \
                                                                                             \
        CacheMode mode = CacheMode::cacheMode;                                               \
        const uint8_t expectedMode = GetCacheModeValue(mode);                                \
        uint8_t actualMode = 99;                                                             \
                                                                                             \
        MOCKER(DataCopyGM2UBImpl, void (*)(T *, T *, const DataCopyParams &, const uint8_t)) \
            .stubs()                                                                         \
            .with(any(), any(), any(), spy(actualMode));                                     \
        MOCKER(DataCopyUB2GMImpl, void (*)(T *, T *, const DataCopyParams &, const uint8_t)) \
            .stubs()                                                                         \
            .with(any(), any(), any(), spy(actualMode));                                     \
        TPipe tpipe;                                                                         \
                                                                                             \
        GlobalTensor<T> globalTensor;                                                        \
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);         \
        globalTensor.SetL2CacheHint(mode);                                                   \
                                                                                             \
        TBuf<TPosition::VECIN> tbuf1;                                                        \
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

DATA_COPY_L2_CACHE_UB_TEST(uint16_t, CACHE_MODE_NORMAL)
DATA_COPY_L2_CACHE_UB_TEST(int8_t, CACHE_MODE_DISABLE)

#define DATA_COPY_L2_CACHE_UB_PAD_TEST(dataType, cacheMode)                                                \
    TEST_F(DataCopyL2CacheTestSuite, testCaseUbPad##dataType##cacheMode)                                   \
    {                                                                                                      \
        using T = dataType;                                                                                \
        uint8_t srcGm[1024]{ 0 };                                                                          \
        uint32_t dataSize = 1024 / sizeof(T);                                                              \
                                                                                                           \
        CacheMode mode = CacheMode::cacheMode;                                                             \
        const uint8_t expectedMode = GetCacheModeValue(mode);                                              \
        uint8_t actualMode = 99;                                                                           \
                                                                                                           \
        MOCKER(DataCopyPadUB2GMImpl, void (*)(T *, T *, const DataCopyParams &, const uint8_t))            \
            .stubs()                                                                                       \
            .with(any(), any(), any(), spy(actualMode));                                                   \
        MOCKER(DataCopyPadUB2GMImpl, void (*)(T *, T *, const DataCopyExtParams &, const uint8_t))         \
            .stubs()                                                                                       \
            .with(any(), any(), any(), spy(actualMode));                                                   \
        MOCKER(DataCopyPadGm2UBImpl,                                                                       \
            void (*)(T *, T *, const DataCopyParams &, const DataCopyPadParams &, const uint8_t))          \
            .stubs()                                                                                       \
            .with(any(), any(), any(), any(), spy(actualMode));                                            \
        MOCKER(DataCopyPadGm2UBImpl,                                                                       \
            void (*)(T *, T *, const DataCopyExtParams &, const DataCopyPadExtParams<T> &, const uint8_t)) \
            .stubs()                                                                                       \
            .with(any(), any(), any(), any(), spy(actualMode));                                            \
        TPipe tpipe;                                                                                       \
                                                                                                           \
        GlobalTensor<T> globalTensor;                                                                      \
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);                       \
        globalTensor.SetL2CacheHint(mode);                                                                 \
                                                                                                           \
        TBuf<TPosition::VECIN> tbuf1;                                                                      \
        tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));                                                     \
        LocalTensor<T> inputLocal = tbuf1.Get<T>();                                                        \
                                                                                                           \
        DataCopyParams param;                                                                              \
        DataCopyPadParams padParam;                                                                        \
        DataCopyExtParams extParam;                                                                        \
        DataCopyPadExtParams<T> padExtParam;                                                               \
                                                                                                           \
        DataCopyPad(inputLocal, globalTensor, param, padParam);                                            \
        EXPECT_EQ(expectedMode, actualMode);                                                               \
                                                                                                           \
        actualMode = 99;                                                                                   \
        DataCopyPad(globalTensor, inputLocal, extParam);                                                   \
        EXPECT_EQ(expectedMode, actualMode);                                                               \
    }

DATA_COPY_L2_CACHE_UB_PAD_TEST(int32_t, CACHE_MODE_NORMAL)
DATA_COPY_L2_CACHE_UB_PAD_TEST(int16_t, CACHE_MODE_DISABLE)

#define DATA_COPY_L2_CACHE_UB_NDDMA_TEST(dataType, cacheMode)                                                          \
    TEST_F(DataCopyL2CacheTestSuite, testCaseUbNDDMA##dataType##cacheMode)                                             \
    {                                                                                                                  \
        using T = dataType;                                                                                            \
        uint8_t srcGm[1024]{ 0 };                                                                                      \
        uint32_t dataSize = 1024 / sizeof(T);                                                                          \
                                                                                                                       \
        CacheMode mode = CacheMode::cacheMode;                                                                         \
        const uint8_t expectedMode = GetCacheModeValue(mode);                                                          \
        uint8_t actualMode = 99;                                                                                       \
        MOCKER(nddma_out_to_ub_b8, void (*)(void *, void *, uint8_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, \
            uint8_t, uint8_t, bool, uint8_t))                                                                          \
            .stubs()                                                                                                   \
            .with(any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), any(), spy(actualMode));       \
        TPipe tpipe;                                                                                                   \
                                                                                                                       \
        GlobalTensor<T> globalTensor;                                                                                  \
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);                                   \
        globalTensor.SetL2CacheHint(mode);                                                                             \
                                                                                                                       \
        TBuf<TPosition::VECIN> tbuf1;                                                                                  \
        tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));                                                                 \
        LocalTensor<T> inputLocal = tbuf1.Get<T>();                                                                    \
                                                                                                                       \
        MultiCopyParams<T, 2> params;                                                                                  \
                                                                                                                       \
        DataCopy(inputLocal, globalTensor, params);                                                                    \
        EXPECT_EQ(expectedMode, actualMode);                                                                           \
    }

DATA_COPY_L2_CACHE_UB_NDDMA_TEST(int8_t, CACHE_MODE_NORMAL)
DATA_COPY_L2_CACHE_UB_NDDMA_TEST(uint8_t, CACHE_MODE_DISABLE)

TEST_F(DataCopyL2CacheTestSuite, testCaseSetGlobalTensorMultipleTimes)
{
    uint8_t srcGm[1024 * sizeof(uint32_t)]{ 0 };
    uint32_t dataSize = 1024;

    const CacheMode mode = CacheMode::CACHE_MODE_DISABLE;
    uint8_t expectedMode = GetCacheModeValue(mode);
    uint8_t actualMode = 99;
    MOCKER(DataCopyGM2UBImpl, void (*)(uint32_t *, uint32_t *, const DataCopyParams &, const uint8_t))
        .stubs()
        .with(any(), any(), any(), spy(actualMode));
    TPipe tpipe;

    GlobalTensor<uint32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(srcGm), dataSize);
    globalTensor.SetL2CacheHint(mode);

    TBuf<TPosition::VECIN> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(uint32_t));
    LocalTensor<uint32_t> inputLocal = tbuf1.Get<uint32_t>();
    DataCopyParams params;

    DataCopy(inputLocal, globalTensor, params);
    EXPECT_EQ(expectedMode, actualMode);

    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(srcGm), dataSize);
    DataCopy(inputLocal, globalTensor, params);
    EXPECT_EQ(expectedMode, actualMode);
}

TEST_F(DataCopyL2CacheTestSuite, testCaseSetL2CacheHintMultipleTimes)
{
    uint8_t srcGm[1024 * sizeof(uint32_t)]{ 0 };
    uint32_t dataSize = 1024;

    CacheMode mode = CacheMode::CACHE_MODE_DISABLE;
    uint8_t actualMode = 99;
    MOCKER(DataCopyGM2UBImpl, void (*)(uint32_t *, uint32_t *, const DataCopyParams &, const uint8_t))
        .stubs()
        .with(any(), any(), any(), spy(actualMode));
    TPipe tpipe;

    GlobalTensor<uint32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(srcGm), dataSize);
    globalTensor.SetL2CacheHint(mode);
    mode = CacheMode::CACHE_MODE_NORMAL;
    uint8_t expectedMode = GetCacheModeValue(mode);
    globalTensor.SetL2CacheHint(mode);

    TBuf<TPosition::VECIN> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(uint32_t));
    LocalTensor<uint32_t> inputLocal = tbuf1.Get<uint32_t>();
    DataCopyParams params;

    DataCopy(inputLocal, globalTensor, params);
    EXPECT_EQ(expectedMode, actualMode);
}

TEST_F(DataCopyL2CacheTestSuite, testCaseSetGlobalTensorWithModifidAddr)
{
    uint8_t srcGm[1024 * sizeof(uint32_t)]{ 0 };
    uint32_t dataSize = 1024;

    CacheMode mode = CacheMode::CACHE_MODE_DISABLE;
    const uint8_t expectedMode = GetCacheModeValue(mode);
    uint8_t actualMode = 99;
    MOCKER(DataCopyGM2UBImpl, void (*)(uint32_t *, uint32_t *, const DataCopyParams &, const uint8_t))
        .stubs()
        .with(any(), any(), any(), spy(actualMode));
    TPipe tpipe;

    GlobalTensor<uint32_t> globalTensor;
    globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(srcGm), dataSize);
    globalTensor.SetL2CacheHint(mode);

    TBuf<TPosition::VECIN> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(uint32_t));
    LocalTensor<uint32_t> inputLocal = tbuf1.Get<uint32_t>();
    DataCopyParams params;

    DataCopy(inputLocal, globalTensor, params);
    EXPECT_EQ(expectedMode, actualMode);

    GlobalTensor<uint32_t> globalTensor2;
    globalTensor2.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(globalTensor.address_), dataSize);
    DataCopy(inputLocal, globalTensor2, params);
    EXPECT_EQ(expectedMode, actualMode);
    EXPECT_EQ(globalTensor2.cacheMode_, mode);
}