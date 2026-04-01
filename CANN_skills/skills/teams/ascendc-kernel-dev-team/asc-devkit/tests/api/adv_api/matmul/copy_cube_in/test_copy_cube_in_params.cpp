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
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_params.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "base_tiling_struct.h"

using namespace std;
using namespace AscendC;

namespace {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);

public:
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;

    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetRuntimeParams(int32_t baseUseK, int32_t baseUseN, int32_t stepKbIdx = 0, int32_t stepNIdx = 0, bool isTranspose = false) {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(),
            tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgN(tiling.GetSingleCoreM());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKb(tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranspose);
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestCopyCubeInParams : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, void> mm;
};

TEST_F(TestCopyCubeInParams, all_interface) {
    TilingParams tilingParams = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    int32_t baseUseK = 96;
    int32_t baseUseN = 48;
    int32_t stepKbIdx = 0;
    int32_t stepNIdx = 1;
    mm.InitVar(tiling);
    mm.SetRuntimeParams(baseUseK, baseUseN, stepKbIdx, stepNIdx);
    // for tmp
    EXPECT_EQ(mm.GetStepCol(), 2);
    EXPECT_EQ(mm.GetStepRow(), 3);
    EXPECT_EQ(mm.GetBufferPos(), stepKbIdx);
    EXPECT_EQ(mm.IsKRowDirec(), true);
    EXPECT_EQ(mm.GetOrgHeight(), 256);
    EXPECT_EQ(mm.GetOrgWidth(), 64);
    EXPECT_EQ(mm.GetBaseHeight(), 96);
    EXPECT_EQ(mm.GetBaseWidth(), 48);
    EXPECT_EQ(mm.GetSingleHeight(), 256);
    EXPECT_EQ(mm.GetSingleWidth(), 48);
    // EXPECT_EQ(mm.GetTotalRow(), 3);
    // EXPECT_EQ(mm.GetTotalCol(), 1);
    EXPECT_EQ(mm.GetBufferSize(), 96 * 64);
    EXPECT_EQ(mm.GetDepth(), 4);
}
