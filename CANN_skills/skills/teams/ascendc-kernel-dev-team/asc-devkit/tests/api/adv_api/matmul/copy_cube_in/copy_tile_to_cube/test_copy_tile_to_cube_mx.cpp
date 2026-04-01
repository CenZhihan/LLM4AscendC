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
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "../base_tiling_struct.h"

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
        return 1;
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
    using CubeInBuffer = CustomCubeInBuffer<IMPL, MatmulInputScaleAType<A_TYPE, fp8_e8m0_t>, MM_CFG>;
    using DataCopyUtilsScaleA = Impl::Detail::CopyTileToCubeWrapper<IMPL, MM_CFG, MatmulInputScaleAType<A_TYPE, fp8_e8m0_t>>;
    using DataCopyUtilsScaleB = Impl::Detail::CopyTileToCubeWrapper<IMPL, MM_CFG, MatmulInputScaleBType<B_TYPE, fp8_e8m0_t>>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleB)
, MATMUL_IMPORT_MODULE(CubeInBuffer)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleB);
    MATMUL_ALLOW_USING(CubeInBuffer);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleB);
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
    using CopyCubeInParams = typename AscendC::Conditional<TAG == InputTypeTag::scaleA, CopyCubeInParamsScaleA, CopyCubeInParamsScaleB>::type;
    template <InputTypeTag TAG>
    using MatmulTensorInfo = typename AscendC::Conditional<TAG == InputTypeTag::scaleA, MatmulTensorInfoScaleA, MatmulTensorInfoScaleB>::type;
    template <InputTypeTag TAG>
    using DataCopyWrapper = typename AscendC::Conditional<TAG == InputTypeTag::scaleA, DataCopyWrapperScaleA, DataCopyWrapperScaleB>::type;

    MATMUL_USE_MODULE(DataCopyUtilsScaleA);
    MATMUL_USE_MODULE(DataCopyUtilsScaleB);
    MATMUL_USE_MODULE(CubeInBuffer);
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
        MATMUL_MODULE(CubeInBuffer)->Init(0, 0);
    }

    void SetRuntimeParams() {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgM(tiling.GetSingleCoreM());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgN(tiling.GetSingleCoreN());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKa(tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKb(tiling.GetSingleCoreK());
    }

    void SetTransposeScale(bool isTransposeScaleA, bool isTransposeScaleB){
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleA(isTransposeScaleA);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleB(isTransposeScaleB);
    }

    void RunCase(int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth) {
        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());

        LocalTensor<SrcT> l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor();
        MATMUL_MODULE(DataCopyUtilsScaleA)->template CopyTileToCube(l1, curRow, curCol, tileHeight, tileWidth);
        MATMUL_MODULE(DataCopyUtilsScaleB)->template CopyTileToCube(l1, curRow, curCol, tileHeight, tileWidth);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    
};
}

class TestCopyTileToCubeMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE_ND = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, AscendC::TPosition::GM, CubeFormat::ND, false, AscendC::TPosition::GM>;
    using AS_TYPE_NZ = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::NZ,
        fp4x2_e2m1_t, false, AscendC::TPosition::GM, CubeFormat::NZ, false, AscendC::TPosition::GM>;

    using BS_TYPE_ND = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, true, AscendC::TPosition::GM, CubeFormat::ND, true, AscendC::TPosition::GM>;
    using BS_TYPE_NZ = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::NZ,
        fp4x2_e1m2_t, true, AscendC::TPosition::GM, CubeFormat::NZ, true, AscendC::TPosition::GM>;

    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE_ND, BS_TYPE_ND, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;
    MatmulImpl<AS_TYPE_NZ, BS_TYPE_NZ, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;
};

TEST_F(TestCopyTileToCubeMx, Copy_Tile_to_Cube_ScaleAB_ND) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 444, 331, 4096, 444, 331, 4096, 256, 256, 128, 32, 64, 1, 2, 32, 32, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.RunCase(0, 0, 32, 64);
}


TEST_F(TestCopyTileToCubeMx, Copy_Tile_to_Cube_ScaleAB_NZ) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 375, 75, 2048, 375, 75, 2048, 384, 96, 64, 32, 32, 1, 1, 32, 32, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetRuntimeParams();
    mm2.RunCase(0, 0, 32, 64);
}


TEST_F(TestCopyTileToCubeMx, Copy_Tile_to_Cube_ScaleAB_Transpose) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 116, 159, 2160, 116, 159, 2160, 128, 160, 128, 17, 17, 1, 1, 17, 17, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.SetTransposeScale(true, false); // by default, ScaleA is not transpose and ScaleB is transpose.
    mm1.RunCase(0, 0, 32, 64);
}
