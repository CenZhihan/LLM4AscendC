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

#include <iostream>
#include <fstream>

namespace AscendC {

namespace {

constexpr uint32_t NCDHW_FractalZ3D = 1;
constexpr uint32_t FractalZ3D_NCDHW = 2;
constexpr uint32_t NCDHW_NDC1HWC0 = 3;
constexpr uint32_t NDC1HWC0_NCDHW = 4;


constexpr TransDataConfig config1 = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
constexpr TransDataConfig config2 = {DataFormat::FRACTAL_Z_3D, DataFormat::NCDHW};
constexpr TransDataConfig config3 = {DataFormat::NCDHW, DataFormat::NDC1HWC0};
constexpr TransDataConfig config4 = {DataFormat::NDC1HWC0, DataFormat::NCDHW};

}

template <typename T, uint32_t mode>
class KernelTransData {
public:
__aicore__ inline KernelTransData() {}
__aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm,
    int32_t n, int32_t c, int32_t d, int32_t h, int32_t w, TPipe *tpipe)
{
    this->d = d;
    this->c = c;
    this->h = h;
    this->w = w;
    this->n = n;
    this->c1 = (c + c0 - 1) / c0;
    this->n1 = (n + n0 - 1) / n0;
    this->hw1 = (h*w + hw0 - 1) / hw0;

    if (mode == NDC1HWC0_NCDHW) {
        this->srcShapeSize = n * c1 * c0 * d * h * w;
        this->dstShapeSize = n * d * c * hw0;
        this->tmpShapeSize = 512 + d * c1 * c0 * hw0 * hw1;
        uint32_t dstGmSize = n * c * d * h * w;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcShapeSize * sizeof(T));
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstGmSize * sizeof(T));
    } else {
        if constexpr (mode == NCDHW_FractalZ3D) {
            srcShapeSize = n * d * c * hw0 * hw1;
            dstShapeSize = n1 * n0 * c1 * c0 * d * h * w;
            if ((h*w) % 16 != 0 ) {
                needPad = true;
                dstShapeSize = n1 * n0 * c1 * c0 * d * hw0 * hw1;
            }
            tmpShapeSize = c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1;
        } else if constexpr (mode == FractalZ3D_NCDHW) {
            this->srcShapeSize = d * c1 * h * w * n1 * n0 * c0;
            this->dstShapeSize = n * c * d * (hw1 * hw0);
            this->tmpShapeSize = d * c1 * (hw1 * hw0) * n1 * n0 * c0 * 2;
        } else if constexpr (mode == NCDHW_NDC1HWC0) {
            this->srcShapeSize = n * d * c * hw0;
            this->dstShapeSize = n * c1 * c0 * d * h * w;
            this->tmpShapeSize = d * hw0 * hw1 + d * c1 * c0 * hw0 * hw1;
        }
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcShapeSize * sizeof(T));
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstShapeSize * sizeof(T));
    }


    this->pipe = tpipe;
    pipe->InitBuffer(inQueue, 1, srcShapeSize * sizeof(T));
    pipe->InitBuffer(outQueue, 1, dstShapeSize * sizeof(T));
    pipe->InitBuffer(tmpBuf, tmpShapeSize * sizeof(T));

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
    LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
    if constexpr (mode == NCDHW_FractalZ3D || mode == NCDHW_NDC1HWC0) {
        DataCopyExtParams extParam = {static_cast<uint16_t>(n * c * d),
            static_cast<uint32_t>(h * w * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParam = {true, 0, 0, 0};
        if (needPad) {
            DataCopyPad(srcLocal, srcGlobal, extParam, padParam);
        } else {
            DataCopy(srcLocal, srcGlobal, srcShapeSize);
        }
    } else if constexpr (mode == FractalZ3D_NCDHW || mode == NDC1HWC0_NCDHW) {
        DataCopy(srcLocal, srcGlobal, srcShapeSize);
    }

    inQueue.EnQue(srcLocal);
}
__aicore__ inline void Compute()
{
    LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
    LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
    LocalTensor<T> srcLocal = inQueue.DeQue<T>();
    PipeBarrier<PIPE_V>();

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout ndc1hwc0Layout = MakeLayout(MakeShape(n, d, c1, h, w, c0), MakeStride());
    Layout fractalLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    if constexpr (mode == NCDHW_FractalZ3D) {
        TransDataParams<decltype(ncdhwLayout), decltype(fractalLayout)> params = {ncdhwLayout, fractalLayout};
        TransData<config1>(dstLocal, srcLocal, tmp, params);
    } else if constexpr (mode == FractalZ3D_NCDHW) {
        TransDataParams<decltype(fractalLayout), decltype(ncdhwLayout)> params = {fractalLayout, ncdhwLayout};
        TransData<config2, T, decltype(fractalLayout), decltype(ncdhwLayout)>(dstLocal, srcLocal, tmp, params);
    } else if constexpr (mode == NCDHW_NDC1HWC0) {
        TransDataParams<decltype(ncdhwLayout), decltype(ndc1hwc0Layout)> params = {ncdhwLayout, ndc1hwc0Layout};
        TransData<config3>(dstLocal, srcLocal, tmp, params);
    } else if constexpr (mode == NDC1HWC0_NCDHW) {
        TransDataParams<decltype(ndc1hwc0Layout), decltype(ncdhwLayout)> params = {ndc1hwc0Layout, ncdhwLayout};
        TransData<config4>(dstLocal, srcLocal, tmp, params);
    }

    outQueue.EnQue<T>(dstLocal);
    inQueue.FreeTensor(srcLocal);

}
__aicore__ inline void CopyOut()
{
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();
    DataCopyExtParams extParam {static_cast<uint16_t>(n * c * d), static_cast<uint32_t>(h*w*sizeof(T)), 0, 0, 0};
    if constexpr (mode == NCDHW_FractalZ3D) {
        DataCopy(dstGlobal, dstLocal, n1 * n0 * c1);
    } else if constexpr (mode == FractalZ3D_NCDHW) {
        DataCopy(dstGlobal, dstLocal, dstShapeSize);
    } else if constexpr (mode == NCDHW_NDC1HWC0) {
        DataCopy(dstGlobal, dstLocal, dstShapeSize);
    } else if constexpr (mode == NDC1HWC0_NCDHW) {
        DataCopyPad(dstGlobal, dstLocal, extParam);
    }
    outQueue.FreeTensor(dstLocal);
}

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    TPipe *pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECIN> tmpBuf;
    bool needPad = false;
    int32_t n = 0;
    int32_t c = 0;
    int32_t d = 0;
    int32_t h = 0;
    int32_t w = 0;
    int32_t n1 = 0;
    int32_t c1 = 0;
    int32_t hw1 = 0;
    int32_t c0 = 16;
    int32_t n0 = 16;
    int32_t hw0 = 16;
    uint32_t srcShapeSize = 0;
    uint32_t dstShapeSize = 0;
    uint32_t tmpShapeSize = 0;
};
} // namespace AscendC

template <typename T, uint32_t mode>
__global__ __aicore__ void MainTransdata(
    __gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint64_t n, uint64_t c, uint64_t d, uint64_t h, uint64_t w)
{
    if (g_coreType == AscendC::AIC || AscendC::GetBlockIdx() > 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::KernelTransData<T, mode> op;
    op.Init(srcGm, dstGm, n, c, d, h, w, &pipe);
    op.Process();
}

struct TransDataTestParams {
    int32_t n;
    int32_t c;
    int32_t d;
    int32_t h;
    int32_t w;
    uint32_t mode;
    void (*cal_func)(uint8_t*, uint8_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
};

class TransDataTestsuite : public testing::Test, public testing::WithParamInterface<TransDataTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPERATTION_TRANSDATA, TransDataTestsuite,
    ::testing::Values(
        TransDataTestParams { 5, 32, 2, 1, 16, 1, MainTransdata<half, 1> },
        TransDataTestParams { 4, 31, 1, 6, 7, 2, MainTransdata<half, 2> },
        TransDataTestParams { 4, 20, 2, 3, 1, 3, MainTransdata<half, 3> },
        TransDataTestParams { 8, 14, 2, 1, 16, 4, MainTransdata<half, 4> },
        TransDataTestParams { 5, 32, 2, 1, 16, 1, MainTransdata<bfloat16_t, 1> },
        TransDataTestParams { 4, 31, 1, 6, 7, 2, MainTransdata<bfloat16_t, 2> },
        TransDataTestParams { 4, 20, 2, 3, 1, 3, MainTransdata<bfloat16_t, 3> },
        TransDataTestParams { 8, 14, 2, 1, 16, 4, MainTransdata<bfloat16_t, 4> }

        ));

TEST_P(TransDataTestsuite, TransDataOpTestCase)
{
    auto params = GetParam();
    auto n = params.n;
    auto c = params.c;
    auto d = params.d;
    auto h = params.h;
    auto w = params.w;
    auto mode = params.mode;
    uint32_t srcShapeSize;
    uint32_t dstShapeSize;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    int32_t c0 = 16;
    int32_t n0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    if (mode == 1) {
        srcShapeSize = n * d * c * hw0 * hw1;
        dstShapeSize = n1 * n0 * c1 * c0 * d * h * w;
        if ((h*w) % 16 != 0 ) {
            dstShapeSize = n1 * n0 * c1 * c0 * d * hw0 * hw1;
        }
    } else if (mode == 2) {
        srcShapeSize = d * c1 * h * w * n1 * n0 * c0;
        dstShapeSize = n * c * d * (hw1 * hw0);
    } else if (mode == 3) {
        srcShapeSize = n * d * c * hw0;
        dstShapeSize = n * c1 * c0 * d * h * w;
    } else if (mode == 4) {
        srcShapeSize = n * c1 * c0 * d * h * w;
        dstShapeSize = n * d * c * hw0;
    }
    uint8_t srcGm[srcShapeSize * sizeof(half)] = {0}; // external guarantee inner is 32B aligned
    uint8_t dstGm[dstShapeSize * sizeof(half)] = {0};
    params.cal_func(dstGm, srcGm, n, c, d, h, w);
    EXPECT_EQ(dstGm[0], 0);
}
