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
#include "kernel_utils.h"
#include "common.h"
#include <math.h>
#include <iostream>
#include <fstream>
using namespace AscendC;
namespace AscendC {
// T, U for l0c->gm
template <typename T, typename U> class KernelDataCopyL0c2Gm {
public:
    __aicore__ inline KernelDataCopyL0c2Gm() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode, uint8_t clipReluPreIn,
        uint8_t elewiseOpIn, bool channelSplitIn, bool nz2ndEnIn)
    {
        srcGlobal.SetGlobalBuffer((__gm__ U*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcCO1, 1, 512 * sizeof(U));
        quantMode = mode;
        clipReluPre = clipReluPreIn;
        elewiseOp = elewiseOpIn;
        channelSplit = channelSplitIn;
        nz2ndEn = nz2ndEnIn;
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        ;
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> srcLocal = inQueueSrcCO1.AllocTensor<U>();

        DataCopyCO12DstParams dataCopyParams(16, 32, 8, 1, quantMode, 0, channelSplit, nz2ndEn);
        dataCopyParams.eltWiseOp = elewiseOp;
        DataCopy(dstGlobal, srcLocal, dataCopyParams);
        inQueueSrcCO1.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        ;
    }

private:
    TPipe pipe;
    TQueBind<TPosition::CO1, TPosition::GM, 1> inQueueSrcCO1;
    GlobalTensor<U> srcGlobal;
    GlobalTensor<T> dstGlobal;
    QuantMode_t quantMode;
    uint8_t clipReluPre;
    uint8_t elewiseOp;
    bool channelSplit;
    bool nz2ndEn;
};
} // namespace AscendC

template <typename T, typename U>
__global__ __aicore__ void MainDataCopyL0c2Gm(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode,
    uint8_t clipReluPre, uint8_t elewiseOp, bool channelSplit, bool nz2ndEn)
{
    AscendC::KernelDataCopyL0c2Gm<T, U> op;
    op.Init(dstGm, srcGm, mode, clipReluPre, elewiseOp, channelSplit, nz2ndEn);
    op.Process();
}

struct DataCopyL0c2GmTestParams {
    int32_t dstTypeSize;
    int32_t srcTypeSize;
    QuantMode_t quantMode;
    void (*cal_func)(uint8_t*, uint8_t*, QuantMode_t, uint8_t, uint8_t, bool, bool);
    uint8_t clipReluPre;
    uint8_t elewiseOp;
    bool channelSplit;
    bool nz2ndEn;
};

class DataCopyL0c2GmTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyL0c2GmTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYL0C2GM, DataCopyL0c2GmTestsuite,
   ::testing::Values(
    DataCopyL0c2GmTestParams { 4, 4, QuantMode_t::NoQuant, MainDataCopyL0c2Gm<int32_t, int32_t>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 4, 4, QuantMode_t::NoQuant, MainDataCopyL0c2Gm<float, float>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::QF322B8_PRE, MainDataCopyL0c2Gm<int8_t, float>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::QF322B8_PRE, MainDataCopyL0c2Gm<uint8_t, float>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::VQF322B8_PRE, MainDataCopyL0c2Gm<int8_t, float>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::F322F16, MainDataCopyL0c2Gm<half, float>, 1, 1, true, true },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::F322BF16, MainDataCopyL0c2Gm<bfloat16_t, float>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::DEQF16, MainDataCopyL0c2Gm<half, int32_t>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::VDEQF16, MainDataCopyL0c2Gm<half, int32_t>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::REQ8, MainDataCopyL0c2Gm<int8_t, int32_t>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::VREQ8, MainDataCopyL0c2Gm<int8_t, int32_t>, 0, 0, false, false },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::VREQ8, MainDataCopyL0c2Gm<uint8_t, int32_t>, 0, 0, false, false }
    ));

TEST_P(DataCopyL0c2GmTestsuite, DataCopyL0c2GmOpTestCase)
{
    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    auto param = GetParam();
    uint8_t srcGm[512 * param.srcTypeSize];
    uint8_t dstGm[512 * param.dstTypeSize];

    param.cal_func(dstGm, srcGm, param.quantMode, param.clipReluPre, param.elewiseOp, param.channelSplit, param.nz2ndEn);
    g_coreType = tmp;
}