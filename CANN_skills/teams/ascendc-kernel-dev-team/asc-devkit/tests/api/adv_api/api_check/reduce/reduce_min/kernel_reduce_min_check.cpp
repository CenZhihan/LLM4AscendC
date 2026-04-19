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
#include "impl/adv_api/detail/api_check/kernel_api_check.h"
#include "../reduce_case_common.h"

class ReduceMinAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        AscendC::SetGCoreType(2);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::AR>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::AR>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::AR>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcInnerPad)
{
    CheckReduceSrcInnerPad<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinSrcPos)
{
    CheckReduceSrcPos<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckReduceMinDstPos)
{
    CheckReduceDstPos<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}


TEST_F(ReduceMinAPICheck, ReduceMinAPICheckTmpPos)
{
    CheckReduceTmpPos<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}

TEST_F(ReduceMinAPICheck, ReduceMinAPICheckOverlap)
{
    CheckReduceOverlap<float, 
        HighLevelApiCheck::CheckFuncReduceMin<float, AscendC::Pattern::Reduce::RA>>("ReduceMin");

}
