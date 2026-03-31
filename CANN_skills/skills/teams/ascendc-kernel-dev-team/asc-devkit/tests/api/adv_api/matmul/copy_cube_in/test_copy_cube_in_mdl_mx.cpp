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
#include "include/adv_api/matmul/constant_tiling.h"
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
    using CubeInBufferA = CustomCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = CustomCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CopyCubeInA = Impl::Detail::CopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = Impl::Detail::CopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeInBufferScaleA)
, MATMUL_IMPORT_MODULE(CubeInBufferScaleB)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleA)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleB)
, MATMUL_IMPORT_MODULE(CubeInBufferA)
, MATMUL_IMPORT_MODULE(CubeInBufferB)
, MATMUL_IMPORT_MODULE(CopyCubeInA)
, MATMUL_IMPORT_MODULE(CopyCubeInB)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoScaleB)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsA)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyCubeInParamsB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyUtilsB)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperA)
, MATMUL_IMPORT_MODULE_PRIVATE(DataCopyWrapperB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoA)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulTensorInfoB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUserDefineInfo)
{
    MATMUL_ALLOW_USING(CubeInBufferScaleA);
    MATMUL_ALLOW_USING(CubeInBufferScaleB);
    MATMUL_ALLOW_USING(CopyCubeInScaleA);
    MATMUL_ALLOW_USING(CopyCubeInScaleB);
    MATMUL_ALLOW_USING(CubeInBufferA);
    MATMUL_ALLOW_USING(CubeInBufferB);
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsScaleB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsScaleB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperScaleA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperScaleB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoScaleA);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoScaleB);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsA);
    MATMUL_ALLOW_USING_PRIVATE(CopyCubeInParamsB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyUtilsB);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperA);
    MATMUL_ALLOW_USING_PRIVATE(DataCopyWrapperB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoA);
    MATMUL_ALLOW_USING_PRIVATE(MatmulTensorInfoB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUserDefineInfo);

    using SrcT = fp8_e8m0_t;
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;

public:
    using VAR_PARAMS = typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using CallBack = MM_CB;

    template <InputTypeTag TAG>
    using CubeInBuffer = typename ConditionalMulti<TAG == InputTypeTag::A || TAG == InputTypeTag::B,
        TAG == InputTypeTag::A || TAG == InputTypeTag::scaleA,
        CubeInBufferA, CubeInBufferB, CubeInBufferScaleA, CubeInBufferScaleB>::type;

    template <InputTypeTag TAG>
    using CopyCubeInParams = typename ConditionalMulti<TAG == InputTypeTag::A || TAG == InputTypeTag::B,
        TAG == InputTypeTag::A || TAG == InputTypeTag::scaleA,
        CopyCubeInParamsA, CopyCubeInParamsB, CopyCubeInParamsScaleA, CopyCubeInParamsScaleB>::type;

    template <InputTypeTag TAG>
    using MatmulTensorInfo = typename ConditionalMulti<TAG == InputTypeTag::A || TAG == InputTypeTag::B,
        TAG == InputTypeTag::A || TAG == InputTypeTag::scaleA,
        MatmulTensorInfoA, MatmulTensorInfoB, MatmulTensorInfoScaleA, MatmulTensorInfoScaleB>::type;

    template <InputTypeTag TAG>
    using DataCopyUtils = typename ConditionalMulti<TAG == InputTypeTag::A || TAG == InputTypeTag::B,
        TAG == InputTypeTag::A || TAG == InputTypeTag::scaleA,
        DataCopyUtilsA, DataCopyUtilsB, DataCopyUtilsScaleA, DataCopyUtilsScaleB>::type;

    template <InputTypeTag TAG>
    using DataCopyWrapper = typename ConditionalMulti<TAG == InputTypeTag::A || TAG == InputTypeTag::B,
        TAG == InputTypeTag::A || TAG == InputTypeTag::scaleA,
        DataCopyWrapperA, DataCopyWrapperB, DataCopyWrapperScaleA, DataCopyWrapperScaleB>::type;

    MATMUL_USE_MODULE(CopyCubeInScaleA);
    MATMUL_USE_MODULE(CopyCubeInScaleB);
    MATMUL_USE_MODULE(CopyCubeInA);
    MATMUL_USE_MODULE(CopyCubeInB);
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
    }

    void SetTranspose(bool isTransposeA, bool isTransposeB, bool isTransposeScaleA, bool isTransposeScaleB){
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleA(isTransposeScaleA);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeScaleB(isTransposeScaleB);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isTransposeA);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTransposeB);
    }

    void RunCase(int32_t curRow, int32_t curCol, int32_t tileM, int32_t tileN, int32_t tileK) {
        int32_t tileScaleK = Ceil(tileK, AscendC::Impl::MX_BASEK_FACTOR) * AscendC::Impl::MX_EVEN_FACTOR;
        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());

        // scaleA
        MATMUL_MODULE(CopyCubeInScaleA)->template Init();
        MATMUL_MODULE(CopyCubeInScaleA)->template LoadData(curRow, curCol, tileM, tileScaleK);
        MATMUL_MODULE(CopyCubeInScaleA)->template ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
        // scaleB
        MATMUL_MODULE(CopyCubeInScaleB)->template Init();
        MATMUL_MODULE(CopyCubeInScaleB)->template LoadData(curRow, curCol, tileScaleK, tileN);
        MATMUL_MODULE(CopyCubeInScaleB)->template ClearLoadData(EMPTY_TENSOR<SrcT>, 0, 0);
        // A
        MATMUL_MODULE(CopyCubeInA)->template Init();
        MATMUL_MODULE(CopyCubeInA)->template LoadData(curRow, curCol, tileM, tileK);
        MATMUL_MODULE(CopyCubeInA)->template ClearLoadData(EMPTY_TENSOR<SrcAT>, 0, 0);
        // B
        MATMUL_MODULE(CopyCubeInB)->template Init();
        MATMUL_MODULE(CopyCubeInB)->template LoadData(curRow, curCol, tileK, tileN);
        MATMUL_MODULE(CopyCubeInB)->template ClearLoadData(EMPTY_TENSOR<SrcBT>, 0, 0);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    
};

class TestCopyCubeInMdlMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;

    using A_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, true, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using B_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;

    using A_FP4_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using B_FP4_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e1m2_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;

    using A_FP8_NZ_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::NZ, false, TPosition::GM>;
    using B_FP8_NZ_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::NZ, true, TPosition::GM>;

    using A_FP4_NZ_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp4x2_e2m1_t, true, TPosition::GM, CubeFormat::NZ, false, TPosition::GM>;
    using B_FP4_NZ_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp4x2_e1m2_t, false, TPosition::GM, CubeFormat::NZ, true, TPosition::GM>;

    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;

    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_MDL;
    constexpr static MatmulShapeParams mShapeParams = {365, 382, 320, 256, 256, 256};

    constexpr static MatmulFuncParams mFncParams{false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, true};
    constexpr static MatmulConfig ORDER_M_CFG = GetMMConfig<configMode>(mShapeParams, mFncParams);
    constexpr static MatmulApiStaticTiling ORDER_M_TILING = GetMatmulApiTiling<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE>(ORDER_M_CFG);
    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, ORDER_M_TILING, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;

    constexpr static MatmulShapeParams nShapeParams = {209, 386, 114, 256, 256, 128};
    constexpr static MatmulFuncParams nFncParams{false, false, false, false, 0, IterateOrder::ORDER_N, ScheduleType::INNER_PRODUCT, true, true};
    constexpr static MatmulConfig ORDER_N_CFG = GetMMConfig<configMode>(nShapeParams, nFncParams);
    constexpr static MatmulApiStaticTiling ORDER_N_TILING = GetMatmulApiTiling<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE>(ORDER_N_CFG);
    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, ORDER_N_TILING, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm3;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm4;
    MatmulImpl<A_FP4_TYPE, B_FP4_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm5;
    MatmulImpl<A_FP8_NZ_TYPE, B_FP8_NZ_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmFp8NZ;
    MatmulImpl<A_FP4_NZ_TYPE, B_FP4_NZ_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmFp4NZ;

    constexpr static MatmulConfig staticMmCfgUnAlign = {
        .doNorm = true,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 128,
        .basicN = 128,
        .basicK = 128,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 102,
        .singleCoreN = 102,
        .singleCoreK = 102,
        .stepM = 1,
        .stepN = 1,
        .baseMN = 4096,
        .singleCoreMN = 4096,
        .enUnitFlag = false,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = BatchMode::SINGLE_LARGE_THAN_L1,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = true
    };
    constexpr static MatmulApiStaticTiling staticTilingUnAlign =
        GetMatmulApiTiling<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE>(staticMmCfgUnAlign);
    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, staticTilingUnAlign,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> padZeroMM;
};

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_IterateOrder_Tiling_M_Config_UNDEF) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 465, 251, 192, 465, 251, 192, 256, 256, 192, 2, 1, 2, 1, 1, 1, 1, 0, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    const int32_t tileM = 465;
    const int32_t tileN = 251;
    const int32_t tileK = 192;
    mm1.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_IterateOrder_Tiling_N_Config_UNDEF) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 454, 30, 128, 454, 30, 128, 480, 32, 128, 1, 1, 1, 1, 1, 1, 1, 1, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.SetTranspose(false, true, false, true);
    const int32_t tileM = 454;
    const int32_t tileN = 30;
    const int32_t tileK = 128;
    mm1.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_IterateOrder_Tiling_UNDEF_Config_M) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 365, 382, 320, 365, 382, 320, 256, 256, 256, 4, 4, 2, 2, 2, 2, 1, 2, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetRuntimeParams();
    mm2.SetTranspose(false, true, false, true);
    const int32_t tileM = 365;
    const int32_t tileN = 382;
    const int32_t tileK = 320;
    mm2.RunCase(0, 0, tileM, tileN, tileK);
}


TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_IterateOrder_Tiling_UNDEF_Config_N) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 209, 386, 114, 209, 386, 114, 224, 256, 128, 1, 2, 1, 2, 1, 1, 1, 2, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm3.InitVar(tiling);
    mm3.SetRuntimeParams();
    mm3.SetTranspose(false, true, false, true);
    const int32_t tileM = 209;
    const int32_t tileN = 256;
    const int32_t tileK = 114;
    mm3.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_FP8_CeilK_Is_Odd) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 382, 2, 400, 382, 2, 400, 192, 32, 256, 4, 2, 2, 1, 2, 2, 0, 0, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm4.InitVar(tiling);
    mm4.SetRuntimeParams();
    mm4.SetTranspose(true, false, false, true);
    const int32_t tileM = 382;
    const int32_t tileN = 2;
    const int32_t tileK = 400;
    mm4.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_FP4_CeilK_Is_Odd) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 356, 442, 466, 356, 442, 466, 128, 256, 256, 6, 4, 3, 2, 2, 2, 0, 0, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm5.InitVar(tiling);
    mm5.SetRuntimeParams();
    mm5.SetTranspose(false, true, false, true);
    const int32_t tileM = 356;
    const int32_t tileN = 442;
    const int32_t tileK = 466;
    mm5.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_FP8_NZ_CeilK_Is_Odd) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 96, 64, 64, 96, 64, 64, 64, 2, 2, 1, 1, 2, 2, 0, 0, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mmFp8NZ.InitVar(tiling);
    mmFp8NZ.SetRuntimeParams();
    mmFp8NZ.SetTranspose(false, false, false, true);
    const int32_t tileM = 64;
    const int32_t tileN = 64;
    const int32_t tileK = 96;
    mmFp8NZ.RunCase(0, 0, tileM, tileN, tileK);
}

TEST_F(TestCopyCubeInMdlMx, Copy_Cube_In_FP4_NZ_CeilK_Is_Odd) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, MxTypePara
    TilingParamsMx tilingParamsMx = {1, 128, 256, 96, 128, 256, 96, 128, 256, 64, 2, 2, 1, 1, 2, 2, 0, 0, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mmFp4NZ.InitVar(tiling);
    mmFp4NZ.SetRuntimeParams();
    mmFp4NZ.SetTranspose(true, false, false, true);
    const int32_t tileM = 128;
    const int32_t tileN = 256;
    const int32_t tileK = 96;
    mmFp4NZ.RunCase(0, 0, tileM, tileN, tileK);
}
}
