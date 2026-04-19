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

class TestCacheSuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestCacheSuite, DataCachePreloadTest) {
    uint64_t src[10] = {0};
    int16_t cacheOffset = 10;
    EXPECT_NO_THROW(AscendC::DataCachePreloadImpl(src, cacheOffset));
}

TEST_F(TestCacheSuite, PreloadImplTest) {
    int64_t preFetchLen = 64;
    void* pc = reinterpret_cast<void*>(0x1000);
    EXPECT_NO_THROW(AscendC::PreLoadImpl(pc, preFetchLen));
}

TEST_F(TestCacheSuite, GetICachePreloadStatusTest) {
    EXPECT_EQ(AscendC::GetICachePreloadStatusImpl(), 0);
}

TEST_F(TestCacheSuite, PreloadTest) {
    int64_t preFetchLen = 64;
    EXPECT_NO_THROW(AscendC::PreLoad(preFetchLen));
}
