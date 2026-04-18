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

class ReduceSumAPICheck : public testing::Test {
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

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::AR>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::AR>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::AR>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcInnerPad)
{
    CheckReduceSrcInnerPad<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumSrcPos)
{
    CheckReduceSrcPos<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");

}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckReduceSumDstPos)
{
    CheckReduceDstPos<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");

}


TEST_F(ReduceSumAPICheck, ReduceSumAPICheckTmpPos)
{
    CheckReduceTmpPos<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");
}

TEST_F(ReduceSumAPICheck, ReduceSumAPICheckOverlap)
{
    CheckReduceOverlap<float, 
        HighLevelApiCheck::CheckFuncReduceSum<float, AscendC::Pattern::Reduce::RA>>("ReduceSum");

}
