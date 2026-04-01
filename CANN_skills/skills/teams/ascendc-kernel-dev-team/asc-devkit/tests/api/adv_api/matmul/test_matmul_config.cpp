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
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"

using namespace std;
using namespace AscendC;


class TestMatmulConfig : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(TestMatmulConfig, TestParamsConfig)
{
    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
    constexpr static MatmulShapeParams shapeParams{128, 128, 128, 64, 64, 64};
    constexpr static MatmulQuantParams quantParams{1, 1};
    constexpr static MatmulBatchParams batchParams{1, BatchMode::BATCH_LARGE_THAN_L1};
    constexpr static MatmulFuncParams funcParams{1, 1, 1, 1, 1, IterateOrder::ORDER_N, ScheduleType::OUTER_PRODUCT,
        1, 1};
    constexpr MatmulConfig mmConfig = GetMMConfig<configMode>(shapeParams, quantParams, batchParams, funcParams);

    EXPECT_EQ(mmConfig.doNorm, true);
    EXPECT_EQ(mmConfig.singleCoreM, 128);
    EXPECT_EQ(mmConfig.singleCoreN, 128);
    EXPECT_EQ(mmConfig.singleCoreK, 128);
    EXPECT_EQ(mmConfig.basicM, 64);
    EXPECT_EQ(mmConfig.basicN, 64);
    EXPECT_EQ(mmConfig.basicK, 64);
    EXPECT_EQ(mmConfig.isPerTensor, true);
    EXPECT_EQ(mmConfig.hasAntiQuantOffset, true);
    EXPECT_EQ(mmConfig.isNBatch, true);
    EXPECT_EQ(mmConfig.batchMode, BatchMode::BATCH_LARGE_THAN_L1);
    EXPECT_EQ(mmConfig.intrinsicsCheck, true);
    EXPECT_EQ(mmConfig.enVecND2NZ, true);
    EXPECT_EQ(mmConfig.enableDoubleCache, true);
    EXPECT_EQ(mmConfig.enableL1CacheUB, true);
    EXPECT_EQ(mmConfig.doMTE2Preload, 1);
    EXPECT_EQ(mmConfig.iterateOrder, IterateOrder::ORDER_N);
    EXPECT_EQ(mmConfig.scheduleType, ScheduleType::OUTER_PRODUCT);
    EXPECT_EQ(mmConfig.enableReuse, true);
    EXPECT_EQ(mmConfig.enableUBReuse, true);
}
