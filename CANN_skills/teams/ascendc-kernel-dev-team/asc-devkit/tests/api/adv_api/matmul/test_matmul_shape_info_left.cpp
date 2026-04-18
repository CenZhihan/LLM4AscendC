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
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/param/matmul_shape_tiling.h"

using namespace std;
using namespace AscendC;


namespace {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUserDefineInfo)
{
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUserDefineInfo);
    using SrcT = typename A_TYPE::T;
public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MatmulImpl() {
        InitVar();
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        tiling.depthA1 = 2;
        tiling.depthB1 = 2;
        tiling.iterateOrder = 0;
        tiling.baseM = 16;
        tiling.baseN = 16;
        tiling.baseK = 16;
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetSingleCoreParams(int32_t singleM, int32_t singleN, int32_t singleK, bool isIntraBlock = false)
    {
        if (isIntraBlock) {
            intraBlockMatmul.singleCoreM = singleM;
            intraBlockMatmul.singleCoreN = singleN;
            intraBlockMatmul.singleCoreK = singleK;
            intraBlockMatmul.M = singleM;
            intraBlockMatmul.N = singleN;
            intraBlockMatmul.Ka = singleK;
            intraBlockMatmul.Kb = singleK;
        } else {
            MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(singleM, singleN, singleK);
            MATMUL_MODULE(MatmulShapeInfo)->SetOrgShape(singleM, singleN, singleK, singleK, singleN);
            MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
            MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
        }
        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
    }

    void SetInitAParams(int32_t stepM, int32_t stepKa, int32_t baseM, int32_t baseK, int32_t depth, bool isIntraBlock = false) {
        tiling.stepM = stepM;
        tiling.stepKa = stepKa;
        tiling.baseM = baseM;
        tiling.baseK = baseK;
        tiling.depthA1 = depth;
        var.baseUseK_ = baseK;
        var.baseUseStepKa_ = baseK * stepKa;
        if (isIntraBlock) {
            intraBlockMatmul.baseUseM = baseM;
            intraBlockMatmul.mIter = intraBlockMatmul.singleCoreM / baseM;
            intraBlockMatmul.kIter = intraBlockMatmul.singleCoreK / baseK;
        } else {
            var.baseUseM_ = baseM;
        }
    }

    void SetALayoutInfo(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d, int32_t batch)
    {
        tiling.ALayoutInfoB = b;
        tiling.ALayoutInfoS = s;
        tiling.ALayoutInfoN = n;
        tiling.ALayoutInfoG = g;
        tiling.ALayoutInfoD = d;
        batchA_ = batch;
    }

    void SetInitBParams(int32_t stepN, int32_t stepKb, int32_t baseN, int32_t baseK, int32_t depth, bool isIntraBlock = false) {
        tiling.stepN = stepN;
        tiling.stepKb = stepKb;
        tiling.baseN = baseN;
        tiling.baseK = baseK;
        tiling.depthB1 = depth;
        var.baseUseK_ = baseK;
        var.baseUseStepKb_ = baseK * stepKb;
        if (isIntraBlock) {
            intraBlockMatmul.baseUseN = baseN;
            intraBlockMatmul.nIter = intraBlockMatmul.singleCoreN / baseN;
            intraBlockMatmul.kIter = intraBlockMatmul.singleCoreK / baseK;
        } else {
            var.baseUseN_ = baseN;
        }
    }

    void SetBLayoutInfo(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d, int32_t batch)
    {
        tiling.BLayoutInfoB = b;
        tiling.BLayoutInfoS = s;
        tiling.BLayoutInfoN = n;
        tiling.BLayoutInfoG = g;
        tiling.BLayoutInfoD = d;
        batchB_ = batch;
    }

    void SetTranspose(bool isATrans, bool isBTrans, bool isIntraBlock = false)
    {
        if (isIntraBlock) {
            intraBlockMatmul.isTransposeA = isATrans;
            intraBlockMatmul.isTransposeB = isBTrans;
        } else {
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isATrans);
            MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isBTrans);
        }
    }

    void SetSelfDefineData(uint64_t dataPtr) {
        MATMUL_MODULE(MatmulUserDefineInfo)->SetSelfDefineData(dataPtr);
    }

    void SetUserDefineInfo(uint64_t tilingPtr) {
        MATMUL_MODULE(MatmulUserDefineInfo)->SetUserDefineInfo(tilingPtr);
    }

    uint32_t GetMIter() {
        return MATMUL_MODULE(MLoop)->GetTotalIter();
    }

    uint32_t GetNIter() {
        return MATMUL_MODULE(NLoop)->GetTotalIter();
    }

    uint32_t GetKIter() {
        return MATMUL_MODULE(KLoop)->GetTotalIter();
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
    int32_t batchA_;
    int32_t batchB_;

    struct IntraBlock {
        __aicore__ inline IntraBlock(){};
        __gm__ SrcT* aGlobal;
        __gm__ SrcT* bGlobal;
        int M;
        int N;
        int Ka;
        int Kb;
        int Kc;
        int singleCoreM;
        int singleCoreN;
        int singleCoreK;
        int mIter;
        int nIter;
        int kIter;
        int baseUseM;
        int baseUseN;
        // measured in cube block
        int blockUseM;
        int blockUseN;
        int tailM, tailK, tailN;
        int cacheProcA = 0;
        bool enableBias = false;
        bool isTransposeA;
        bool isTransposeB;
        bool fakeMsg = false;
    };

    IntraBlock intraBlockMatmul;
};
}

class TestMatmulShapeInfoA : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void> mm;
};

TEST_F(TestMatmulShapeInfoA, test_get_singleCore_params)
{
    mm.SetSingleCoreParams(32, 32, 16);
    EXPECT_EQ(mm.GetSingleCoreM(), 32);
    EXPECT_EQ(mm.GetSingleCoreN(), 32);

    EXPECT_EQ(mm.GetKIter(), 1);
    EXPECT_EQ(mm.GetMIter(), 2);
    EXPECT_EQ(mm.GetNIter(), 2);

    EXPECT_EQ(mm.GetOrgKa(), 16);
    EXPECT_EQ(mm.GetOrgKb(), 16);
    EXPECT_EQ(mm.GetOrgKc(), 32);
    EXPECT_EQ(mm.GetOrgN(), 32);
    EXPECT_EQ(mm.GetOrgM(), 32);
}
