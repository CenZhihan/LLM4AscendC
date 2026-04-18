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
#include "include/adv_api/matmul/matmul.h"
#include "copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;

namespace {
class CustomMatmulScheduler {
public:
    __aicore__ inline void Init(const TCubeTiling *__restrict cubeTiling, TPipe *tpipe) {}
    __aicore__ inline void GetResult(const LocalTensor<float>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false) {}
    __aicore__ inline bool ScheduleOnce(bool enPartialSum) {
        return false;
    }
    __aicore__ inline void Reset() {}
    __aicore__ inline void End() {}
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulWithScalePolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using Scheduler = CustomMatmulScheduler;
};
}

class TestMxMatmulImpl : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e1m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using AS_TYPE_UB = MatmulTypeWithScale<TPosition::GM, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using BS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE_UB = MatmulTypeWithScale<TPosition::GM, TPosition::VECOUT, CubeFormat::ND,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using C_TYPE_GM = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using C_TYPE_UB = MatmulType<TPosition::VECIN, CubeFormat::NZ, float>;

    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE_GM, BS_TYPE_GM, C_TYPE_GM, BIAS_TYPE, CFG_NORM,
               MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm1;
    MatmulImpl<AS_TYPE_UB, BS_TYPE_UB, C_TYPE_UB, BIAS_TYPE, CFG_NORM,
               MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;
};

TEST_F(TestMxMatmulImpl, test_mx_matmul_impl_scaleGm_OutputGm) {
    TPipe pipe;
    TCubeTiling tiling;

    bool isTransposeA = false;
    bool isTransposeAS = false;
    bool isTransposeB = false;
    bool isTransposeBS = false;

    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    tilingParamsMx.GetTiling(tiling);

    using A_T = typename AS_TYPE_GM::T;
    using B_T = typename BS_TYPE_GM::T;
    using C_T = typename C_TYPE_GM::T;
    typedef fp8_e8m0_t MX_T;

    const int32_t left_data_size = tiling.M * tiling.Ka;
    const int32_t left_scale_data_size = tiling.M * ((tiling.Ka + 31) / 32);
    const int32_t right_data_size = tiling.Kb * tiling.N;
    const int32_t right_scale_data_size = ((tiling.Kb + 31) / 32) * tiling.N;
    const int32_t output_data_size = tiling.M * tiling.N;

    uint8_t left_global[left_data_size * sizeof(A_T)] = {0};
    uint8_t left_scale_global[left_scale_data_size * sizeof(MX_T)] = {0};
    uint8_t right_global[right_data_size * sizeof(B_T)] = {0};
    uint8_t right_scale_global[right_scale_data_size * sizeof(MX_T)] = {0};
    uint8_t output_global[output_data_size * sizeof(C_T)] = {0};

    GM_ADDR aGM = left_global;
    GM_ADDR amxGM = left_scale_global;
    GM_ADDR bGM = right_global;
    GM_ADDR bmxGM = right_scale_global;
    GM_ADDR cGM = output_global;

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<MX_T> amxGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<MX_T> bmxGlobal;
    GlobalTensor<C_T> cGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), left_data_size);
    amxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ MX_T*>(amxGM), left_scale_data_size);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), right_data_size);
    bmxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ MX_T*>(bmxGM), right_scale_data_size);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM), output_data_size);

    auto gmA = aGlobal[0];
    auto gmAMX = amxGlobal[0];
    auto gmB = bGlobal[0];
    auto gmBMX = bmxGlobal[0];
    auto gmC = cGlobal[0];

    mm1.Init(&tiling, &pipe);

    mm1.SetTensorA(gmA, isTransposeA);
    mm1.SetTensorScaleA(gmAMX, isTransposeAS);
    mm1.SetTensorB(gmB, isTransposeB);
    mm1.SetTensorScaleB(gmBMX, isTransposeBS);
    mm1.IterateAll(gmC);

    mm1.GetMatrixL1Addr();
    mm1.End();
}

TEST_F(TestMxMatmulImpl, test_mx_matmul_impl_scaleUb_OutputUb) {
    TPipe pipe;
    TCubeTiling tiling;

    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    tilingParamsMx.GetTiling(tiling);

    bool isTransposeA = false;
    bool isTransposeAS = false;
    bool isTransposeB = false;
    bool isTransposeBS = false;

    using A_T = typename AS_TYPE_UB::T;
    using B_T = typename BS_TYPE_UB::T;
    using C_T = typename C_TYPE_UB::T;
    typedef fp8_e8m0_t MX_T;

    const int32_t left_data_size = tiling.M * tiling.Ka;
    const int32_t left_scale_data_size = tiling.M * ((tiling.Ka + 31) / 32);
    const int32_t right_data_size = tiling.Kb * tiling.N;
    const int32_t right_scale_data_size = ((tiling.Kb + 31) / 32) * tiling.N;
    const int32_t output_data_size = tiling.M * tiling.N;

    uint8_t left_global[left_data_size * sizeof(A_T)] = {0};
    uint8_t left_scale_global[left_scale_data_size * sizeof(MX_T)] = {0};
    uint8_t right_global[right_data_size * sizeof(B_T)] = {0};
    uint8_t right_scale_global[right_scale_data_size * sizeof(MX_T)] = {0};
    uint8_t output_global[output_data_size * sizeof(C_T)] = {0};

    GM_ADDR aGM = left_global;
    GM_ADDR bGM = right_global;
    TQue<TPosition::VECOUT, 1> leftMatrixScale;
    TQue<TPosition::VECOUT, 1> rightMatrixScale;
    TQue<TPosition::VECIN, 1> resultCMatrix;

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    LocalTensor<MX_T> bufferMxLeft;
    LocalTensor<MX_T> bufferMxRight;
    LocalTensor<C_T> bufferC;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), left_data_size);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), right_data_size);
    pipe.InitBuffer(leftMatrixScale, 1, left_scale_data_size * sizeof(MX_T));
    pipe.InitBuffer(rightMatrixScale, 1, right_scale_data_size * sizeof(MX_T));
    pipe.InitBuffer(resultCMatrix, 1, output_data_size * sizeof(C_T));

    auto gmA = aGlobal[0];
    auto gmB = bGlobal[0];
    bufferMxLeft = leftMatrixScale.AllocTensor<MX_T>();
    bufferMxRight = rightMatrixScale.AllocTensor<MX_T>();
    bufferC = resultCMatrix.AllocTensor<C_T>();

    mm2.Init(&tiling, &pipe);

    mm2.SetTensorA(gmA, isTransposeA);
    mm2.SetTensorScaleA(bufferMxLeft, isTransposeAS);
    mm2.SetTensorB(gmB, isTransposeB);
    mm2.SetTensorScaleB(bufferMxRight, isTransposeBS);
    mm2.IterateAll(bufferC);

    //  mm2.GetMatrixL1Addr();
    KfcSetIntraScaleAId(mm2, 0);
    KfcSetIntraScaleBId(mm2, 0);
    mm2.End();
}
