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
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "base_tiling_struct.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/base/copy_cube_in_using_ub.h"

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
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using CubeInBufferA = CustomCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = CustomCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = Impl::Detail::CopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = Impl::Detail::CopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeInBufferA)
, MATMUL_IMPORT_MODULE(CubeInBufferB)
, MATMUL_IMPORT_MODULE(CopyCubeInA)
, MATMUL_IMPORT_MODULE(CopyCubeInB)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulSubBlockInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(LocalWorkspace)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperB)
{
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING(CubeInBufferA);
    MATMUL_ALLOW_USING(CubeInBufferB);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoA);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulSubBlockInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(LocalWorkspace);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperB);

    using SrcT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;

    template <InputTypeTag TAG>
    using CubeInBuffer = typename AscendC::Conditional<TAG == InputTypeTag::A, CubeInBufferA, CubeInBufferB>::type;
    template <InputTypeTag TAG>
    using CopyCubeInParams = typename AscendC::Conditional<TAG == InputTypeTag::A, CopyCubeInParamsA, CopyCubeInParamsB>::type;

    template <InputTypeTag TAG>
    using MatmulTensorInfo = typename AscendC::Conditional<TAG == InputTypeTag::A, MatmulTensorInfoA, MatmulTensorInfoB>::type;

    template <InputTypeTag TAG>
    using DataCopyUtils = typename AscendC::Conditional<TAG == InputTypeTag::A, DataCopyUtilsA, DataCopyUtilsB>::type;

    template <InputTypeTag TAG>
    using DataCopyWrapper =
        typename AscendC::Conditional<TAG == InputTypeTag::A, DataCopyWrapperA, DataCopyWrapperB>::type;

    using CallBack = MM_CB;

    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MatmulImpl() {}

    VAR_PARAMS& GetVar()
    {
        return var;
    }

    void InitLocalWorkspace()
    {
        pipe.InitBuffer(ubWorkspace, 2, 81920);
        workspaceBuffer = ubWorkspace.AllocTensor<uint8_t>();
        MATMUL_MODULE(LocalWorkspace)->Init(workspaceBuffer);
    }

    void InitVar(const TCubeTiling &tiling)
    {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
        InitLocalWorkspace();
    }

    void SetRuntimeParams(int32_t baseUseM, int32_t baseUseK)
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(),
            tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgM(tiling.GetSingleCoreM());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgN(tiling.GetSingleCoreN());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKa(tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKb(tiling.GetSingleCoreK());
        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
    }

    void SetGMInput(bool isTransposeA = false, bool isTransposeB = false)
    {
        GlobalTensor<SrcT> fakeInputA;
        GlobalTensor<SrcBT> fakeInputB;
        MATMUL_MODULE(CopyCubeInA)->SetInput(fakeInputA, isTransposeA);
        MATMUL_MODULE(CopyCubeInB)->SetInput(fakeInputB, isTransposeB);
    }

    void SetUBInput(bool isTransposeA = false, bool isTransposeB = false)
    {
        pipe.InitBuffer(leftMatrix, 1, MATMUL_MODULE(MatmulShapeInfo)->GetOrgKa() * MATMUL_MODULE(MatmulShapeInfo)->GetOrgM());
        pipe.InitBuffer(rightMatrix, 1, MATMUL_MODULE(MatmulShapeInfo)->GetOrgKb() * MATMUL_MODULE(MatmulShapeInfo)->GetOrgN());
        LocalTensor<SrcT> fakeInputA;
        LocalTensor<SrcBT> fakeInputB;
        fakeInputA = leftMatrix.AllocTensor<SrcT>();
        fakeInputB = rightMatrix.AllocTensor<SrcBT>();
        MATMUL_MODULE(CopyCubeInA)->SetInput(fakeInputA, isTransposeA);
        MATMUL_MODULE(CopyCubeInB)->SetInput(fakeInputB, isTransposeB);
    }

    void RunCase(int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth,
        bool isTransposeA = false, bool isTransposeB = false, bool isUBIn = false)
    {
        MATMUL_MODULE(CopyCubeInA)->Init();
        MATMUL_MODULE(CopyCubeInB)->Init();
        if (isUBIn) {
            SetUBInput(isTransposeA);
        } else {
            SetGMInput(isTransposeB);
        }
        auto tensorA = MATMUL_MODULE(CopyCubeInA)->template LoadData(curRow, curCol, tileHeight, tileWidth);
        auto tensorB = MATMUL_MODULE(CopyCubeInB)->template LoadData(curRow, curCol, tileHeight, tileWidth);
        MATMUL_MODULE(CopyCubeInA)->ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
        MATMUL_MODULE(CopyCubeInB)->ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
        MATMUL_MODULE(CopyCubeInA)->Destroy();
        MATMUL_MODULE(CopyCubeInB)->Destroy();
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    int32_t batchA_;
    int32_t batchB_;
    TQue<TPosition::VECIN, 2> leftMatrix;
    TQue<TPosition::VECIN, 2> rightMatrix;
    TQue<TPosition::VECIN, 2> ubWorkspace;
    LocalTensor<uint8_t> workspaceBuffer;
};
}

class TestCopyCubeInMDL310P : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE_ND = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using A_TYPE_NZ = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, half, false>;
    using A_TYPE_VECTOR = MatmulType<AscendC::TPosition::GM, CubeFormat::VECTOR, half, false>;

    using A_TYPE_UB_ND = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, half, false>;
    using A_TYPE_UB_NZ = MatmulType<AscendC::TPosition::VECIN, CubeFormat::NZ, half, false>;
    using A_TYPE_UB_VECTOR = MatmulType<AscendC::TPosition::VECIN, CubeFormat::VECTOR, half, false>;

    using A_TYPE_INT8_NOTRANS = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, false>;
    using B_TYPE_INT8_NOTRANS = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, false>;
    using B_TYPE_INT8_TRANS = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, true>;

    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    // Copy_GM_ND_Half_NoTrans
    MatmulImpl<A_TYPE_ND, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
    // Copy_GM_NZ_Half_NoTrans
    MatmulImpl<A_TYPE_NZ, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;
    // Copy_GM_VECTOR_Half_NoTrans
    MatmulImpl<A_TYPE_VECTOR, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm3;

    // Copy_UB_ND_Half_NoTrans
    MatmulImpl<A_TYPE_UB_ND, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm4;
    // Copy_UB_NZ_Half_NoTrans
    MatmulImpl<A_TYPE_UB_NZ, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm5;
    // Copy_UB_VECTOR_Half_NoTrans
    MatmulImpl<A_TYPE_UB_VECTOR, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm6;

    // Copy_B_Int8_No_Trans
    MatmulImpl<A_TYPE_INT8_NOTRANS, B_TYPE_INT8_NOTRANS, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm_b_int8_no_trans;
    // Copy_A16W8
    MatmulImpl<A_TYPE_ND, B_TYPE_INT8_NOTRANS, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm_a16w8;
};

TEST_F(TestCopyCubeInMDL310P, Copy_GM_ND_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    int32_t mIter = 2;
    int32_t kIter = 3;
    mm.InitVar(tiling);
    mm.SetRuntimeParams(32, 64);
    mm.RunCase(0, 0, 32, 64, false, false, false);
}

TEST_F(TestCopyCubeInMDL310P, Copy_GM_NZ_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetRuntimeParams(32, 64);
    mm2.RunCase(0, 0, 32, 64, false, false, false);
}

TEST_F(TestCopyCubeInMDL310P, Copy_GM_VECTOR_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm3.InitVar(tiling);
    mm3.SetRuntimeParams(32, 64);
    mm3.RunCase(0, 0, 32, 64, false, false, false);
}


TEST_F(TestCopyCubeInMDL310P, Copy_UB_ND_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    int32_t mIter = 2;
    int32_t kIter = 3;
    mm4.InitVar(tiling);
    mm4.SetRuntimeParams(32, 64);
    mm4.RunCase(0, 0, 32, 64, false, false, true);
}

TEST_F(TestCopyCubeInMDL310P, Copy_UB_NZ_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm5.InitVar(tiling);
    mm5.SetRuntimeParams(32, 64);
    mm5.RunCase(0, 0, 32, 64, false, false, true);
}

TEST_F(TestCopyCubeInMDL310P, Copy_UB_VECTOR_Half_NoTrans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm6.InitVar(tiling);
    mm6.SetRuntimeParams(32, 64);
    mm6.RunCase(0, 0, 32, 64, false, false, true);
}

TEST_F(TestCopyCubeInMDL310P, Copy_B_Int8_No_Trans) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm_b_int8_no_trans.InitVar(tiling);
    mm_b_int8_no_trans.SetRuntimeParams(32, 64);
    mm_b_int8_no_trans.RunCase(0, 0, 32, 64, false, false, false);
}
