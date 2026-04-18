/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_matmul_client.cpp
 * \brief
 */

#define ENABLE_CV_COMM_VIA_SSBUF true // kfc 310 mode

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/matmul_intf.h"
#include "../copy_cube_in/base_tiling_struct.h"

using namespace std;
namespace AscendC {
class TestMatmulClient : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using AS_TYPE_UB_fp4 = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using BS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE_UB_fp4 = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp4x2_e2m1_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using C_TYPE_UB = MatmulType<TPosition::VECIN, CubeFormat::NZ, float>;

    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<AS_TYPE_UB, BS_TYPE_UB, C_TYPE_UB, BIAS_TYPE, CFG_NORM,
               MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm0;
    Matmul<AS_TYPE_UB_fp4, BS_TYPE_UB_fp4, C_TYPE_UB, BIAS_TYPE, CFG_NORM,
               MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm1;
};

TEST_F(TestMatmulClient, test_mx_matmul_client_fp8) {
    TPipe pipe;
    TCubeTiling tiling;

    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    tilingParamsMx.GetTiling(tiling);

    using MX_T = fp8_e8m0_t;

    const int32_t left_scale_data_size = tiling.M * ((tiling.Ka + 31) / 32) * 32;
    const int32_t right_scale_data_size = ((tiling.Kb + 31) / 32) * 32 * tiling.N;

    LocalTensor<MX_T> bufferMxLeft;
    LocalTensor<MX_T> bufferMxRight;
    LocalTensor<uint8_t> tmpBuf;
    TQue<TPosition::VECOUT, 1> tmpBufQue;
    TQue<TPosition::VECOUT, 1> leftMatrixScale;
    TQue<TPosition::VECOUT, 1> rightMatrixScale;

    int32_t tmpBufSize = 16384;
    pipe.InitBuffer(tmpBufQue, 1, tmpBufSize);
    tmpBuf = tmpBufQue.AllocTensor<uint8_t>();
    pipe.InitBuffer(leftMatrixScale, 1, left_scale_data_size * sizeof(MX_T));
    pipe.InitBuffer(rightMatrixScale, 1, right_scale_data_size * sizeof(MX_T));

    bufferMxLeft = leftMatrixScale.AllocTensor<MX_T>();
    bufferMxRight = rightMatrixScale.AllocTensor<MX_T>();
    mm0.SetLocalWorkspace(tmpBuf);

    TBuffAddr tbufOutTmp;
    KfcMsg kfcMsg;
    tbufOutTmp.logicPos = (uint8_t)(TPosition::B1);
    tbufOutTmp.dataLen = Ceil(tiling.Ka / NUM_THIRTYTWO, BLOCK_CUBE) * BLOCK_CUBE * 
        Ceil(tiling.N, BLOCK_CUBE) * BLOCK_CUBE * sizeof(MX_T);
    tbufOutTmp.bufferAddr = kfcMsg.body.quantScalar;
    tbufOutTmp.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(TPosition::B1)) + tbufOutTmp.bufferAddr;
    LocalTensor<MX_T> l1Matrix;
    l1Matrix.SetAddr(tbufOutTmp);
    int32_t c0Size_ = 32;
    int32_t scaleK = Ceil(tiling.Ka, AscendC::Impl::MX_BASEK_FACTOR) * 2;
    int32_t offset = 0;
    mm0.ND2ScaleZZ(reinterpret_cast<LocalTensor<uint8_t>&>(l1Matrix), reinterpret_cast<LocalTensor<uint8_t>&>(bufferMxLeft), tiling.M, Ceil(scaleK, c0Size_) * c0Size_, scaleK, offset);
    mm0.CopyUbNZ2NZForScale(reinterpret_cast<LocalTensor<uint8_t>&>(l1Matrix), reinterpret_cast<LocalTensor<uint8_t>&>(bufferMxLeft), scaleK, Ceil(tiling.M, BLOCK_CUBE) * BLOCK_CUBE, offset);
}

TEST_F(TestMatmulClient, test_mx_matmul_client_fp8_ab) {
    TPipe pipe;
    TCubeTiling tiling;

    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    tilingParamsMx.GetTiling(tiling);

    const int32_t left_data_size = tiling.M * tiling.Ka * sizeof(fp8_e4m3fn_t);
    const int32_t right_data_size = tiling.N * tiling.Kb * sizeof(fp8_e4m3fn_t);

    LocalTensor<fp8_e4m3fn_t> bufferLeft;
    LocalTensor<fp8_e4m3fn_t> bufferRight;
    LocalTensor<uint8_t> tmpBuf;
    TQue<TPosition::VECOUT, 1> tmpBufQue;
    TQue<TPosition::VECOUT, 1> leftMatrix;
    TQue<TPosition::VECOUT, 1> rightMatrix;

    int32_t tmpBufSize = 16384;
    pipe.InitBuffer(tmpBufQue, 1, tmpBufSize);
    tmpBuf = tmpBufQue.AllocTensor<uint8_t>();
    pipe.InitBuffer(leftMatrix, 1, left_data_size * sizeof(fp8_e4m3fn_t));
    pipe.InitBuffer(rightMatrix, 1, right_data_size * sizeof(fp8_e4m3fn_t));

    bufferLeft = leftMatrix.AllocTensor<fp8_e4m3fn_t>();
    bufferRight = rightMatrix.AllocTensor<fp8_e4m3fn_t>();
    mm0.SetLocalWorkspace(tmpBuf);

    KfcMsg kfcMsg;
    TBuffAddr tbufOutTmp2;
    tbufOutTmp2.logicPos = (uint8_t)(TPosition::A1);
    tbufOutTmp2.dataLen = Ceil(tiling.Ka, BLOCK_CUBE) * BLOCK_CUBE * 
        Ceil(tiling.N, BLOCK_CUBE) * BLOCK_CUBE * sizeof(fp8_e4m3fn_t);
    tbufOutTmp2.bufferAddr = kfcMsg.body.quantScalar;
    tbufOutTmp2.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(TPosition::A1)) + tbufOutTmp2.bufferAddr;
    LocalTensor<fp8_e4m3fn_t> AL1Matrix;
    AL1Matrix.SetAddr(tbufOutTmp2);
    int32_t c0Size_ = 32;
    mm0.c0Size_ = c0Size_;
    mm0.CopyUbAToL1ForND(AL1Matrix, bufferLeft, false);
    mm0.CopyUbBToL1ForND(AL1Matrix, bufferRight, true);
    mm0.CopyUbAToL1ForND(AL1Matrix, bufferLeft, false);
    mm0.CopyUbBToL1ForND(AL1Matrix, bufferRight, true);
}

TEST_F(TestMatmulClient, test_mx_matmul_client_fp4) {
    TPipe pipe;
    TCubeTiling tiling;

    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    tilingParamsMx.GetTiling(tiling);

    using MX_T = fp8_e8m0_t;

    const int32_t left_scale_data_size = tiling.M * ((tiling.Ka + 31) / 32) * 32;

    LocalTensor<MX_T> bufferMxLeft;
    LocalTensor<uint8_t> tmpBuf;
    TQue<TPosition::VECOUT, 1> tmpBufQue;
    TQue<TPosition::VECOUT, 1> leftMatrixScale;

    int32_t tmpBufSize = 16384;
    pipe.InitBuffer(tmpBufQue, 1, tmpBufSize);
    tmpBuf = tmpBufQue.AllocTensor<uint8_t>();
    pipe.InitBuffer(leftMatrixScale, 1, left_scale_data_size * sizeof(MX_T));

    bufferMxLeft = leftMatrixScale.AllocTensor<MX_T>();
    mm1.SetLocalWorkspace(tmpBuf);

    KfcMsg kfcMsg;
    TBuffAddr tbufOutTmp;
    tbufOutTmp.logicPos = (uint8_t)(TPosition::A1);
    tbufOutTmp.dataLen = Ceil(tiling.Ka / NUM_THIRTYTWO, BLOCK_CUBE) * BLOCK_CUBE * 
        Ceil(tiling.N, BLOCK_CUBE) * BLOCK_CUBE * sizeof(MX_T);
    tbufOutTmp.bufferAddr = kfcMsg.body.quantScalar;
    tbufOutTmp.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(TPosition::A1)) + tbufOutTmp.bufferAddr;
    LocalTensor<MX_T> l1Matrix;
    l1Matrix.SetAddr(tbufOutTmp);
    mm1.SetLocalWorkspace(tmpBuf);

    int32_t fp4C0Size_ = 64;
    int32_t offset = 0;
    mm1.c0Size_ = fp4C0Size_;
    int32_t scaleK = Ceil(tiling.Ka, AscendC::Impl::MX_BASEK_FACTOR) * 2;
    mm1.ND2ScaleZZ(reinterpret_cast<LocalTensor<uint8_t>&>(l1Matrix), reinterpret_cast<LocalTensor<uint8_t>&>(bufferMxLeft), tiling.N, Ceil(scaleK, fp4C0Size_) * fp4C0Size_, scaleK, offset);
    mm1.CopyUbNZ2NZForScale(reinterpret_cast<LocalTensor<uint8_t>&>(l1Matrix), reinterpret_cast<LocalTensor<uint8_t>&>(bufferMxLeft), scaleK, Ceil(tiling.N, BLOCK_CUBE) * BLOCK_CUBE, offset);
}
}
