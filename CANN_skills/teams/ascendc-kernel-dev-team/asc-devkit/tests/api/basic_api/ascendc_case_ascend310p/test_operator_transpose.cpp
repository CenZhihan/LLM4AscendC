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
#include "mockcpp/mockcpp.hpp"
using namespace std;
using namespace AscendC;

namespace AscendC {
template <typename T>
class KernelTranspose {
public:
    __aicore__ inline KernelTranspose() { }

    __aicore__ inline void Init(__gm__ uint8_t *srcGm, __gm__ uint8_t *dstGm, uint32_t nIn, uint32_t cIn, uint32_t hIn,
        uint32_t wIn, TransposeType transposetypeIn, uint32_t dataSizeIn)
    {
        n = nIn;
        c = cIn;
        h = hIn;
        w = wIn;

        inputSize = (dataSizeIn == 0) ? n * c * h * w : dataSizeIn;

        transposetype = transposetypeIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, inputSize * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, inputSize * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, inputSize);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();

        LocalTensor<uint8_t> stackBuffer;
        PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);

        TransposeParamsExt transposeParams;
        transposeParams.nSize = n;
        transposeParams.cSize = c;
        transposeParams.hSize = h;
        transposeParams.wSize = w;
        transposeParams.transposeType = transposetype;
        Transpose(dstLocal, srcLocal, stackBuffer, transposeParams);
        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, inputSize);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;

    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    uint32_t n, c, h, w, inputSize;
    TransposeType transposetype;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void Transpose4D(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t nIn,
        uint32_t cIn, uint32_t hIn, uint32_t wIn, TransposeType transposetypeIn, uint32_t dataSizeIn)
{
    AscendC::KernelTranspose<T> op;
    op.Init(dstGm, srcGm, nIn, cIn, hIn, wIn, transposetypeIn, dataSizeIn);
    op.Process();
}

struct Transpose4dTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, TransposeType, uint32_t);
    uint32_t nIn;
    uint32_t cIn;
    uint32_t hIn;
    uint32_t wIn;
    TransposeType transposetypeIn;
    uint32_t dataSize = 0;
    bool expectRes = true;
    int32_t errorTimes = 0;
};

class Transpose4dTestsuite : public testing::Test,
    public testing::WithParamInterface<Transpose4dTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_TRANSPOSE, Transpose4dTestsuite,
    ::testing::Values(
    Transpose4dTestParams { 2, Transpose4D<half>, 3, 1, 2, 8, TransposeType::TRANSPOSE_NCHW2NHWC},
    Transpose4dTestParams { 2, Transpose4D<int16_t>, 1, 3, 2, 8, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 2, Transpose4D<uint16_t>, 3, 17, 2, 8, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 1, Transpose4D<int8_t>, 1, 3, 1, 32, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 1, Transpose4D<uint8_t>, 3, 3, 1, 32, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 3, 1, 8, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 4, Transpose4D<int32_t>, 3, 3, 1, 8, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 4, Transpose4D<uint32_t>, 3, 17, 2, 8, TransposeType::TRANSPOSE_NCHW2NHWC },
    Transpose4dTestParams { 2, Transpose4D<half>, 3, 3, 2, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 2, Transpose4D<int16_t>, 1, 3, 2, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 2, Transpose4D<uint16_t>, 3, 17, 2, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 1, Transpose4D<int8_t>, 1, 3, 1, 32, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 1, Transpose4D<uint8_t>, 3, 3, 1, 32, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 3, 1, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 4, Transpose4D<uint32_t>, 3, 3, 1, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    Transpose4dTestParams { 4, Transpose4D<int32_t>, 3, 17, 2, 8, TransposeType::TRANSPOSE_NHWC2NCHW },
    // ND2ND_B16 hSize and wSize must be 16
    Transpose4dTestParams { 4, Transpose4D<float>, 16, 16, 5, 16, TransposeType::TRANSPOSE_ND2ND_B16, 16, false, 1},  // hSize != 16
    Transpose4dTestParams { 4, Transpose4D<float>, 16, 16, 16, 7, TransposeType::TRANSPOSE_ND2ND_B16, 16, false, 1},  // wSize != 16
    // ND2ND_B16 must be uint16
    Transpose4dTestParams { 2, Transpose4D<uint16_t>, 50, 50, 16, 16, TransposeType::TRANSPOSE_ND2ND_B16, 256, true},
    Transpose4dTestParams { 4, Transpose4D<float>, 50, 50, 16, 16, TransposeType::TRANSPOSE_ND2ND_B16, 256, false, 1},
    // csize should be in range [0, 4095]
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 4096, 2, 4, TransposeType::TRANSPOSE_NHWC2NCHW, 32768, false, 2}, // cSize = 4096
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 4092, 2, 4, TransposeType::TRANSPOSE_NHWC2NCHW, 32736, true},
    // h * w * sizeof(dtype) should be 32B align
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 16, 2, 3, TransposeType::TRANSPOSE_NHWC2NCHW, 96, false, 1},
    // h * w in range [0, 4095]
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 2, 64, 64, TransposeType::TRANSPOSE_NHWC2NCHW, 8192, false, 1}, // h * w = 4096
    Transpose4dTestParams { 4, Transpose4D<float>, 1, 2, 63, 64, TransposeType::TRANSPOSE_NHWC2NCHW, 8064, true},
    // test tensor size overflow
    Transpose4dTestParams { 2, Transpose4D<uint16_t>, 50, 50, 16, 16, TransposeType::TRANSPOSE_ND2ND_B16, 256, true},
    Transpose4dTestParams { 2, Transpose4D<uint16_t>, 50, 50, 16, 16, TransposeType::TRANSPOSE_ND2ND_B16, 128, false, 1},  // 预期要512B
    Transpose4dTestParams { 4, Transpose4D<float>, 3, 2, 4, 2, TransposeType::TRANSPOSE_NHWC2NCHW, 32, false, 5}, // 不满足N*c*h*w 预期要48个元素
    Transpose4dTestParams { 4, Transpose4D<float>, 3, 2, 4, 2, TransposeType::TRANSPOSE_NHWC2NCHW, 48, true },
    Transpose4dTestParams { 1, Transpose4D<int8_t>, 8, 4, 8, 4, TransposeType::TRANSPOSE_NCHW2NHWC, 960, false, 1}, // 预期要1024个元素
    Transpose4dTestParams { 1, Transpose4D<uint8_t>, 8, 4, 8, 4, TransposeType::TRANSPOSE_NCHW2NHWC, 1024, true}
));

TEST_P(Transpose4dTestsuite, Transpose4dTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.nIn * param.cIn * param.hIn * param.wIn * param.typeSize] = {0};
    uint8_t dstGm[param.nIn * param.cIn * param.hIn * param.wIn * param.typeSize] = {0};
    if (!param.expectRes) {
        MOCKER(raise, int(*)(int)).times(param.errorTimes).will(returnValue(0));
    }
    param.cal_func(dstGm, srcGm, param.nIn, param.cIn, param.hIn, param.wIn, param.transposetypeIn, param.dataSize);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}