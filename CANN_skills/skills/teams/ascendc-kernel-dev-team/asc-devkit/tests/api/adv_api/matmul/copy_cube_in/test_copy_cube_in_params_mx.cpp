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
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "base_tiling_struct.h"

using namespace std;
using namespace AscendC;

namespace {

struct CaseResult {
    int32_t bufferPos;
    bool isTranspose;
    int32_t bufferSize;
    int32_t depth;
    int32_t scaleFactor;
    bool isKRowDirec;
    int32_t orgHeightT;
    int32_t orgHeightF;
    int32_t orgWidthT;
    int32_t orgWidthF;
    int32_t singleHeightT;
    int32_t singleHeightF;
    int32_t singleWidthT;
    int32_t singleWidthF;
    int32_t baseHeightT;
    int32_t baseHeightF;
    int32_t baseWidthT;
    int32_t baseWidthF;
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
    MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

public:
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using VAR_PARAMS = typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;

    MATMUL_USE_MODULE(CopyCubeInParamsScaleA);
    MATMUL_USE_MODULE(CopyCubeInParamsScaleB);
    MATMUL_USE_MODULE(MLoop);
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

    void SetRuntimeParams() {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgM(tiling.GetSingleCoreM());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgN(tiling.GetSingleCoreN());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKa(tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKb(tiling.GetSingleCoreK());

        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
    }

    void SetTranspose() {
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(true);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleA(false);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(true);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleB(false);
    }

    void RunScaleACase(const CaseResult& caseresult) {
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBufferPos()), caseresult.bufferPos);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template IsTranspose()), caseresult.isTranspose);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBufferSize()), caseresult.bufferSize);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetDepth()), caseresult.depth);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetScaleFactor()), caseresult.scaleFactor);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template IsKRowDirec()), caseresult.isKRowDirec);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetOrgHeight<true, true>()), caseresult.orgHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetOrgHeight<false, false>()), caseresult.orgHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetOrgWidth<true, true>()), caseresult.orgWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetOrgWidth<false, false>()), caseresult.orgWidthF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetSingleHeight<true>()), caseresult.singleHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetSingleHeight<false>()), caseresult.singleHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetSingleWidth<true>()), caseresult.singleWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetSingleWidth<false>()), caseresult.singleWidthF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBaseHeight<true>()), caseresult.baseHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBaseHeight<false>()), caseresult.baseHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBaseWidth<true>()), caseresult.baseWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleA)->template GetBaseWidth<false>()), caseresult.baseWidthF);
    }

    void RunScaleBCase(const CaseResult& caseresult) {
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBufferPos()), caseresult.bufferPos);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template IsTranspose()), caseresult.isTranspose);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBufferSize()), caseresult.bufferSize);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetDepth()), caseresult.depth);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetScaleFactor()), caseresult.scaleFactor);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template IsKRowDirec()), caseresult.isKRowDirec);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetOrgHeight<true, true>()), caseresult.orgHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetOrgHeight<false, false>()), caseresult.orgHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetOrgWidth<true, true>()), caseresult.orgWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetOrgWidth<false, false>()), caseresult.orgWidthF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetSingleHeight<true>()), caseresult.singleHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetSingleHeight<false>()), caseresult.singleHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetSingleWidth<true>()), caseresult.singleWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetSingleWidth<false>()), caseresult.singleWidthF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBaseHeight<true>()), caseresult.baseHeightT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBaseHeight<false>()), caseresult.baseHeightF);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBaseWidth<true>()), caseresult.baseWidthT);
        EXPECT_EQ((MATMUL_MODULE(CopyCubeInParamsScaleB)->template GetBaseWidth<false>()), caseresult.baseWidthF);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestCopyCubeInParamsMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
     using AS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using AS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE_GM, BS_TYPE_GM, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm1;
    MatmulImpl<AS_TYPE_UB, BS_TYPE_UB, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm2;
    MatmulImpl<AS_TYPE_GM, BS_TYPE_GM, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm3;
    MatmulImpl<AS_TYPE_UB, BS_TYPE_UB, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm4;
};

TEST_F(TestCopyCubeInParamsMx, scaleAB_NORM_GM) {
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    CaseResult caseResultA = {0, 0, 96, 2, 1, 0, 8, 64, 64, 8, 8, 64, 64, 8, 3, 32, 32, 3};
    CaseResult caseResultB = {0, 1, 144, 4, 1, 0, 48, 8, 8, 48, 48, 8, 8, 48, 48, 3, 3, 48};

    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.RunScaleACase(caseResultA);
    mm1.RunScaleBCase(caseResultB);
}

TEST_F(TestCopyCubeInParamsMx, scaleAB_NORM_GM_Transpose) {
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    CaseResult caseResultA = {0, 0, 96, 2, 1, 0, 8, 64, 64, 8, 8, 64, 64, 8, 3, 32, 32, 3};
    CaseResult caseResultB = {0, 0, 144, 4, 1, 1, 48, 8, 8, 48, 48, 8, 8, 48, 48, 3, 3, 48};

    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.SetTranspose();
    mm1.RunScaleACase(caseResultA);
    mm1.RunScaleBCase(caseResultB);
}

TEST_F(TestCopyCubeInParamsMx, scaleAB_NORM_UB) {
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16908287};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    CaseResult caseResultA = {0, 0, 2048, 2, 127, 0, 16, 64, 64, 8, 8, 64, 64, 8, 3, 32, 32, 3};
    CaseResult caseResultB = {0, 1, 1024, 4, 127, 0, 48, 8, 32, 48, 48, 8, 8, 48, 48, 3, 3, 48};

    mm2.InitVar(tiling);
    mm2.SetRuntimeParams();
    mm2.RunScaleACase(caseResultA);
    mm2.RunScaleBCase(caseResultB);
}

TEST_F(TestCopyCubeInParamsMx, scaleAB_MDL_GM) {
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16843263};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    CaseResult caseResultA = {0, 0, 96, 2, 127, 0, 8, 64, 64, 8, 8, 64, 64, 8, 3, 32, 32, 3};
    CaseResult caseResultB = {0, 1, 144, 4, 1, 0, 48, 8, 8, 48, 48, 8, 8, 48, 48, 3, 3, 48};

    mm3.InitVar(tiling);
    mm3.SetRuntimeParams();
    mm3.RunScaleACase(caseResultA);
    mm3.RunScaleBCase(caseResultB);
}

TEST_F(TestCopyCubeInParamsMx, scaleAB_MDL_UB) {
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16908033};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    CaseResult caseResultA = {0, 0, 2048, 2, 1, 0, 16, 64, 64, 8, 8, 64, 64, 8, 3, 32, 32, 3};
    CaseResult caseResultB = {0, 1, 1024, 4, 127, 0, 48, 8, 32, 48, 48, 8, 8, 48, 48, 3, 3, 48};

    mm4.InitVar(tiling);
    mm4.SetRuntimeParams();
    mm4.RunScaleACase(caseResultA);
    mm4.RunScaleBCase(caseResultB);
}

