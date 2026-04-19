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
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/copy_cube_in.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "base_tiling_struct.h"

using namespace std;
using namespace AscendC;


namespace {

template <typename T>
const LocalTensor<T> EMPTY_TENSOR;
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class CustomCubeInBuffer {
    using SrcT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CustomCubeInBuffer() {}
    __aicore__ inline ~CustomCubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        GetTPipePtr()->InitBuffer(qid_, 1, 16384);
    }
    __aicore__ inline void Destroy() {
        if (cacheProc_ > 0) {
            qid_.FreeTensor(tensor_);
            cacheProc_ = 0;
        }
        qid_.FreeAllEvent();
    }
    __aicore__ inline int32_t GetIterIndex(int32_t curRow, int32_t curCol)
    {
        return 0;
    }
    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return false;
    }
    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return EMPTY_TENSOR<SrcT>;
    }
    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t bufferPos = -1)
    {
        tensor_ = qid_.template AllocTensor<SrcT>();
        cacheProc_ = 1;
        return tensor_;
    }
    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = EMPTY_TENSOR<SrcT>)
    {
        if (cacheProc_ > 0) {
            qid_.FreeTensor(tensor_);
            cacheProc_ = 0;
        }
    }
    __aicore__ inline int32_t GetBufferHeadAddr()
    {
        return 0;
    }
    __aicore__ inline void Reset()
    {
        FreeTensor();
    }
    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor)
    {
        qid_.EnQue(tensor);
    }
    __aicore__ inline void DeQue() {
        qid_.DeQue();
    }

private:
    TQueBind<TPosition::GM, INPUT_TYPE::TAG == InputTypeTag::A ? TPosition::A1 : TPosition::B1,
        1, GetNdNzMask(CubeFormat::NZ, INPUT_TYPE::format)> qid_;
    LocalTensor<SrcT> tensor_;
    int32_t cacheProc_ = 0;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulWithScalePolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CubeInBufferScaleA = CustomCubeInBuffer<IMPL, MatmulInputScaleAType<A_TYPE, fp8_e8m0_t>, MM_CFG>;
    using CubeInBufferScaleB = CustomCubeInBuffer<IMPL, MatmulInputScaleBType<B_TYPE, fp8_e8m0_t>, MM_CFG>;
    using CopyCubeInScaleA = Impl::Detail::CopyCubeIn<IMPL, MatmulInputScaleAType<A_TYPE, fp8_e8m0_t>, MM_CFG>;
    using CopyCubeInScaleB = Impl::Detail::CopyCubeIn<IMPL, MatmulInputScaleBType<B_TYPE, fp8_e8m0_t>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeInBufferScaleA)
, MATMUL_IMPORT_MODULE(CubeInBufferScaleB)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleA)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeInBufferScaleA);
    MATMUL_ALLOW_USING(CubeInBufferScaleB);
    MATMUL_ALLOW_USING(CopyCubeInScaleA);
    MATMUL_ALLOW_USING(CopyCubeInScaleB);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperScaleA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperScaleB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoScaleA);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoScaleB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

    using SrcT = fp8_e8m0_t;

public:
    using VAR_PARAMS = typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using CallBack = MM_CB;

    template <InputTypeTag TAG>
    using CubeInBuffer = typename Conditional<TAG == InputTypeTag::scaleA, CubeInBufferScaleA, CubeInBufferScaleB>::type;
    template <InputTypeTag TAG>
    using CopyCubeInParams = typename Conditional<TAG == InputTypeTag::scaleA, CopyCubeInParamsScaleA, CopyCubeInParamsScaleB>::type;
    template <InputTypeTag TAG>
    using MatmulTensorInfo = typename Conditional<TAG == InputTypeTag::scaleA, MatmulTensorInfoScaleA, MatmulTensorInfoScaleB>::type;
    template <InputTypeTag TAG>
    using DataCopyUtils = typename Conditional<TAG == InputTypeTag::scaleA, DataCopyUtilsScaleA, DataCopyUtilsScaleB>::type;
    template <InputTypeTag TAG>
    using DataCopyWrapper = typename Conditional<TAG == InputTypeTag::scaleA, DataCopyWrapperScaleA, DataCopyWrapperScaleB>::type;

    MATMUL_USE_MODULE(CopyCubeInScaleA);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(MLoop);
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
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKa(tiling.GetSingleCoreK());
    }

    void RunCase(int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth) {
        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());

        // scaleA
        MATMUL_MODULE(CopyCubeInScaleA)->template Init();
        MATMUL_MODULE(CopyCubeInScaleA)->template LoadData(curRow, curCol, tileHeight, tileWidth);
        MATMUL_MODULE(CopyCubeInScaleA)->template ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
        // scaleB
        MATMUL_MODULE(CopyCubeInScaleB)->template Init();
        MATMUL_MODULE(CopyCubeInScaleB)->template LoadData(curRow, curCol, tileHeight, tileWidth);
        MATMUL_MODULE(CopyCubeInScaleB)->template ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    
};
}

class TestCopyCubeInNormMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using AS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp4x2_e1m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using BS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;

    constexpr static MatmulConfig ORDER_M_CFG = GetNormalConfig(false, false, false, BatchMode::NONE, true, IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, false);
    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, ORDER_M_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;

    constexpr static MatmulConfig ORDER_N_CFG = GetNormalConfig(false, false, false, BatchMode::NONE, true, IterateOrder::ORDER_N, ScheduleType::INNER_PRODUCT, true, false);
    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, ORDER_N_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm3;

    MatmulImpl<AS_TYPE_UB, BS_TYPE_UB, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm4;
};

TEST_F(TestCopyCubeInNormMx, Copy_Cube_In_IterateOrder_Tiling_M_Config_UNDEF) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 304, 1, 320, 304, 1, 320, 256, 32, 256, 4, 2, 2, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.RunCase(0, 0, 32, 64);
}

TEST_F(TestCopyCubeInNormMx, Copy_Cube_In_IterateOrder_Tiling_N_Config_UNDEF) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 448, 80, 192, 448, 80, 192, 448, 96, 128, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.RunCase(0, 0, 32, 64);
}

TEST_F(TestCopyCubeInNormMx, Copy_Cube_In_IterateOrder_Tiling_UNDEF_Config_M) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 48, 288, 192, 48, 288, 192, 48, 288, 192, 1, 1, 1, 1, 1, 1, 1, 2, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetRuntimeParams();
    mm2.RunCase(0, 0, 32, 64);
}


TEST_F(TestCopyCubeInNormMx, Copy_Cube_In_IterateOrder_Tiling_UNDEF_Config_N) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 134, 296, 192, 134, 296, 192, 144, 320, 192, 1, 1, 1, 1, 1, 1, 1, 2, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    mm3.InitVar(tiling);
    mm3.SetRuntimeParams();
    mm3.RunCase(0, 0, 32, 64);
}

TEST_F(TestCopyCubeInNormMx, Copy_Cube_In_IterateOrder_Tiling_M_Config_UNDEF_UB) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 304, 1, 320, 304, 1, 320, 256, 32, 256, 4, 2, 2, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm4.InitVar(tiling);
    mm4.SetRuntimeParams();
    mm4.RunCase(0, 0, 32, 64);
}
