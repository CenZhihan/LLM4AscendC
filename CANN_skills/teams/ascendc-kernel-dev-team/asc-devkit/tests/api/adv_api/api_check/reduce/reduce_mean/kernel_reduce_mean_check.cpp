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


class ReduceMeanAPICheck : public testing::Test {
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

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::AR>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::AR>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::AR>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcInnerPad)
{
    CheckReduceSrcInnerPad<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanSrcPos)
{
    CheckReduceSrcPos<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckReduceMeanDstPos)
{
    CheckReduceDstPos<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}


TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckTmpPos)
{
    CheckReduceTmpPos<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}

TEST_F(ReduceMeanAPICheck, ReduceMeanAPICheckOverlap)
{
    CheckReduceOverlap<float, 
        HighLevelApiCheck::CheckFuncReduceMean<float, AscendC::Pattern::Reduce::RA>>("ReduceMean");

}
