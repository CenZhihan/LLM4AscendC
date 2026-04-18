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

using namespace std;

template <typename srcType, int32_t dim, int32_t brcType=1>
class KernelBroadcast {
public:
    __aicore__ inline KernelBroadcast()
    {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t srcSize_, uint32_t dstSize_)
    {
        const int alginSize = AscendC::GetDataBlockSizeInBytes() / sizeof(srcType);
        srcSize = (srcSize_ + alginSize - 1) / alginSize * alginSize;
        dstSize = (dstSize_ + alginSize - 1) / alginSize * alginSize;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst_gm), dstSize);
        pipe.InitBuffer(inQueue, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(srcType));
    }
    __aicore__ inline void Process(const uint32_t srcShape[dim], const uint32_t dstShape[dim])
    {
        CopyIn();
        Compute(srcShape, dstShape);
        CopyOut();
    }

private:
    __aicore__ inline void Compute(const uint32_t srcShape[dim], const uint32_t dstShape[dim])
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueue.DeQue<srcType>();

        if constexpr (brcType == 1) {
            AscendC::BroadCast<srcType, dim, 0>(dstLocal, srcLocal, dstShape, srcShape);
        } else if constexpr (brcType == 2) {
            // test sharedTmpBuffer
            AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
            AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(sharedTmpBuffer);
            AscendC::BroadCast<srcType, dim, 0>(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
        } else {
            AscendC::BroadcastTiling tiling;
            AscendC::GetBroadcastTilingInfo<srcType>(dim, dstShape, srcShape, false, tiling);
            AscendC::Broadcast(dstLocal, srcLocal, dstShape, srcShape, &tiling);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<srcType> srcLocal = inQueue.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcSize);
        inQueue.EnQue<srcType>(srcLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> srcGlobal;
    AscendC::GlobalTensor<srcType> dstGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

    uint32_t srcSize;
    uint32_t dstSize;
};

template <typename T, int dim>
__global__ __aicore__ void MainKernelBroadcast(uint8_t *dstGm, uint8_t *srcGm, uint32_t srcShape[dim],
    uint32_t dstShape[dim])
{
    uint32_t srcSize = 1;
    uint32_t dstSize = 1;
    for (int i = 0; i < dim; i++) {
        srcSize *= srcShape[i];
        dstSize *= dstShape[i];
    }
    if constexpr((sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint16_t)) && dim <= 2) {
        KernelBroadcast<T, dim, 1> op;
        op.Init(srcGm, dstGm, srcSize, dstSize);
        op.Process(srcShape, dstShape);

        KernelBroadcast<T, dim, 2> op2;
        op2.Init(srcGm, dstGm, srcSize, dstSize);
        op2.Process(srcShape, dstShape);
    }

    KernelBroadcast<T, dim, 3> op3;
    op3.Init(srcGm, dstGm, srcSize, dstSize);
    op3.Process(srcShape, dstShape);
}

template <int dim> struct BroadcastTestParams {
    void (*cal_func)(uint8_t *, uint8_t *, uint32_t *, uint32_t *);
    uint32_t srcSize;
    uint32_t dstSize;
    uint32_t srcShape[dim];
    uint32_t dstShape[dim];
};

template <int dim>
class BroadcastTestsuite : public testing::Test, public testing::WithParamInterface<BroadcastTestParams<dim>> {};

using BroadcastTestsuite_dim1 = BroadcastTestsuite<1>;

INSTANTIATE_TEST_CASE_P(TEST_Broadcast, BroadcastTestsuite_dim1,
    ::testing::Values(BroadcastTestParams<1>{ MainKernelBroadcast<half, 1>, 1, 128, { 1 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<half, 1>, 128, 128, { 128 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<half, 1>, 129, 129, { 129 }, { 129 } },

    BroadcastTestParams<1>{ MainKernelBroadcast<uint8_t, 1>, 1, 128, { 1 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<uint8_t, 1>, 128, 128, { 128 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<uint8_t, 1>, 129, 129, { 129 }, { 129 } },

    BroadcastTestParams<1>{ MainKernelBroadcast<float, 1>, 1, 128, { 1 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<float, 1>, 128, 128, { 128 }, { 128 } },
    BroadcastTestParams<1>{ MainKernelBroadcast<float, 1>, 129, 129, { 129 }, { 129 } }));

TEST_P(BroadcastTestsuite_dim1, BroadcastTestCase)
{
    auto param = GetParam();
    auto &srcShape = param.srcShape;
    auto &dstShape = param.dstShape;
    uint8_t srcGm[param.srcSize] = {0};
    uint8_t dstGm[param.dstSize] = {0};

    param.cal_func(dstGm, srcGm, srcShape, dstShape);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

using BroadcastTestsuite_dim2 = BroadcastTestsuite<2>;

INSTANTIATE_TEST_CASE_P(TEST_Broadcast, BroadcastTestsuite_dim2,
    ::testing::Values(BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 2, 123 * 2, { 1, 2 }, { 123, 2 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 63, 128 * 63, { 1, 63 }, { 128, 63 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 128, 128 * 128, { 1, 128 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 129, 128 * 129, { 1, 129 }, { 128, 129 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 256, 128 * 256, { 1, 256 }, { 128, 256 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 1 * 257, 128 * 257, { 1, 257 }, { 128, 257 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 128 * 129, 128 * 129, { 128, 129 }, { 128, 129 } },

    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 2 * 1, 2 * 29, { 2, 1 }, { 2, 29 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 2 * 1, 2 * 127, { 2, 1 }, { 2, 127 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 63 * 1, 63 * 128, { 63, 1 }, { 63, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 65 * 1, 65 * 129, { 65, 1 }, { 65, 129 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 65 * 1, 65 * 257, { 65, 1 }, { 65, 257 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 65 * 1, 65 * 256, { 65, 1 }, { 65, 256 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 65 * 1, 65 * 288, { 65, 1 }, { 65, 288 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 128 * 1, 128 * 128, { 128, 1 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 129 * 1, 129 * 128, { 129, 1 }, { 129, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<uint8_t, 2>, 129 * 128, 129 * 128, { 129, 128 }, { 129, 128 } },

    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 2, 21 * 2, { 1, 2 }, { 21, 2 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 63, 128 * 63, { 1, 63 }, { 128, 63 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 64, 128 * 64, { 1, 64 }, { 128, 64 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 65, 128 * 65, { 1, 65 }, { 128, 65 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 128, 128 * 128, { 1, 128 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 144, 128 * 144, { 1, 144 }, { 128, 144 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 1 * 129, 128 * 129, { 1, 129 }, { 128, 129 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 128 * 129, 128 * 129, { 128, 129 }, { 128, 129 } },

    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 2 * 1, 2 * 31, { 2, 1 }, { 2, 31 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 9 * 1, 9 * 21, { 9, 1 }, { 9, 21 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 2 * 1, 2 * 65, { 2, 1 }, { 2, 65 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 63 * 1, 63 * 128, { 63, 1 }, { 63, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 64 * 1, 64 * 128, { 64, 1 }, { 64, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 65 * 1, 65 * 128, { 65, 1 }, { 65, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 128 * 1, 128 * 128, { 128, 1 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 129 * 1, 129 * 128, { 129, 1 }, { 129, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<half, 2>, 129 * 128, 129 * 128, { 129, 128 }, { 129, 128 } },

    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 2, 31 * 2, { 1, 2 }, { 31, 2 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 2, 128 * 2, { 1, 2 }, { 128, 2 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 8, 2 * 8, { 1, 8 }, { 2, 8 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 8, 25 * 8, { 1, 8 }, { 25, 8 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 16, 2 * 16, { 1, 16 }, { 2, 16 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 16, 25 * 16, { 1, 16 }, { 25, 16 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 24, 2 * 24, { 1, 24 }, { 2, 24 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 24, 25 * 24, { 1, 24 }, { 25, 24 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 32, 12 * 32, { 1, 32 }, { 12, 32 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 63, 128 * 63, { 1, 63 }, { 128, 63 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 64, 128 * 64, { 1, 64 }, { 128, 64 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 65, 128 * 65, { 1, 65 }, { 128, 65 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 128, 128 * 128, { 1, 128 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 1 * 129, 128 * 129, { 1, 129 }, { 128, 129 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 128 * 129, 128 * 129, { 128, 129 }, { 128, 129 } },

    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 2 * 1, 2 * 8, { 2, 1 }, { 2, 8 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 2 * 1, 2 * 12, { 2, 1 }, { 2, 12 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 25 * 1, 25 * 18, { 25, 1 }, { 25, 18 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 63 * 1, 63 * 34, { 63, 1 }, { 63, 34 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 63 * 1, 63 * 65, { 63, 1 }, { 63, 65 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 64 * 1, 64 * 128, { 64, 1 }, { 64, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 65 * 1, 65 * 128, { 65, 1 }, { 65, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 128 * 1, 128 * 128, { 128, 1 }, { 128, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 129 * 1, 129 * 128, { 129, 1 }, { 129, 128 } },
    BroadcastTestParams<2>{ MainKernelBroadcast<float, 2>, 129 * 128, 129 * 128, { 129, 128 }, { 129, 128 } }));

TEST_P(BroadcastTestsuite_dim2, BroadcastTestCase)
{
    auto param = GetParam();
    auto &srcShape = param.srcShape;
    auto &dstShape = param.dstShape;
    uint8_t srcGm[param.srcSize] = {0};
    uint8_t dstGm[param.dstSize] = {0};

    param.cal_func(dstGm, srcGm, srcShape, dstShape);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

using BroadcastTestsuite_dim3 = BroadcastTestsuite<3>;

INSTANTIATE_TEST_CASE_P(TEST_Broadcast, BroadcastTestsuite_dim3,
    ::testing::Values(BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 2, 2 * 6 * 2, {2, 1, 2 }, {2, 6, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 5, 2 * 28 * 5, {2, 1, 5 }, {2, 28, 5 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 25, 2 * 28 * 25, {2, 1, 25 }, {2, 28, 25 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 63, 2 * 128 * 63, {2, 1, 63 }, {2, 128, 63 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 128, 2 * 128 * 128, {2, 1, 128 }, {2, 128, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 129, 2 * 128 * 129, {2, 1, 129 }, {2, 128, 129 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 256, 2 * 128 * 256, {2, 1, 256 }, {2, 128, 256 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 1 * 257, 2 * 128 * 257, {2, 1, 257 }, {2, 128, 257 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 2 * 128 * 129, 2 * 128 * 129, {2, 128, 129 }, {2, 128, 129 } },

    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 2 * 1, 2 * 2 * 29, {1, 2, 1 }, {2, 2, 29 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 5 * 1, 2 * 5 * 29, {1, 5, 1 }, {2, 5, 29 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 3 * 1, 2 * 3 * 127, {1, 3, 1 }, {2, 3, 127 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 63 * 1, 2 * 63 * 128, {1, 63, 1 }, {2, 63, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 65 * 1, 2 * 65 * 129, {1, 65, 1 }, {2, 65, 129 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 65 * 1, 2 * 65 * 257, {1, 65, 1 }, {2, 65, 257 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 65 * 1, 2 * 65 * 256, {1, 65, 1 }, {2, 65, 256 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 65 * 1, 2 * 65 * 288, {1, 65, 1 }, {2, 65, 288 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 128 * 1, 2 * 128 * 128, {1, 128, 1 }, {2, 128, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 129 * 1, 2 * 129 * 128, {1, 129, 1 }, {2, 129, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<uint8_t, 3>, 1 * 129 * 128, 2 * 129 * 128, {1, 129, 128 }, {2, 129, 128 } },

    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 2, 2 * 21 * 2, {2, 1, 2 }, {2, 21, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 23 * 1 * 2, 23 * 21 * 2, {23, 1, 2 }, {23, 21, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 63, 2 *128 * 63, {2, 1, 63 }, {2, 128, 63 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 64, 2 *128 * 64, {2, 1, 64 }, {2, 128, 64 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 65, 2 *128 * 65, {2, 1, 65 }, {2, 128, 65 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 128, 2 * 128 * 128, {2, 1, 128 }, {2, 128, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 144, 2 * 128 * 144, {2, 1, 144 }, {2, 128, 144 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 1 * 129, 2 * 128 * 129, {2, 1, 129 }, {2, 128, 129 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 2 * 128 * 129, 2 * 128 * 129, {2, 128, 129 }, {2, 128, 129 } },

    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 2 * 1, 2 * 2 * 31, {1, 2, 1 }, {2, 2, 31 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 5 * 1, 2 * 5 * 21, {1, 5, 1 }, {2, 5, 21 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 2 * 1, 2 * 2 * 21, {1, 9, 1 }, {2, 9, 21 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 2 * 1, 2 * 2 * 65, {1, 2, 1 }, {2, 2, 65 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 63 * 1, 2 * 63 * 128, {1, 63, 1 }, {2, 63, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 64 * 1, 2 * 64 * 128, {1, 64, 1 }, {2, 64, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 65 * 1, 2 * 65 * 128, {1, 65, 1 }, {2, 65, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 128 * 1, 2 * 128 * 128, {1, 128, 1 }, {2, 128, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 129 * 1, 2 * 129 * 128, {1, 129, 1 }, {2, 129, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<half, 3>, 1 * 129 * 128, 2 * 129 * 128, {1, 129, 128 }, {2, 129, 128 } },

    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 2, 2 * 11 * 2, {2, 1, 2 }, {2, 11, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 2, 2 * 31 * 2, {2, 1, 2 }, {2, 31, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 2, 2 * 128 * 2, {2, 1, 2 }, {2, 128, 2 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 8, 2 * 2 * 8, {2, 1, 8 }, {2, 2, 8 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 8, 2 * 5 * 8, {2, 1, 8 }, {2, 5, 8 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 8, 2 * 25 * 8, {2, 1, 8 }, {2, 25, 8 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 16, 2 * 2 * 16, {2, 1, 16 }, {2, 2, 16 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 16, 2 * 25 * 16, {2, 1, 16 }, {2, 25, 16 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 24, 2 * 2 * 24, {2, 1, 24 }, {2, 2, 24 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 24, 2 * 25 * 24, {2, 1, 24 }, {2, 25, 24 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 32, 2 * 12 * 32, {2, 1, 32 }, {2, 12, 32 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 63, 2 * 128 * 63, {2, 1, 63 }, {2, 128, 63 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 64, 2 * 128 * 64, {2, 1, 64 }, {2, 128, 64 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 65, 2 * 128 * 65, {2, 1, 65 }, {2, 128, 65 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 128, 2 * 12 * 128, {2, 1, 128 }, {2, 12, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 1 * 129, 2 * 12 * 129, {2, 1, 129 }, {2, 12, 129 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 2 * 12 * 129, 2 * 12 * 129, {2, 12, 129 }, {2, 12, 129 } },

    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 8 * 1, 2 * 8 * 8, {1, 8, 1 }, {2, 8, 8 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 2 * 1, 2 * 2 * 12, {1, 2, 1 }, {2, 2, 12 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 4 * 1, 2 * 4 * 12, {1, 4, 1 }, {2, 4, 12 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 25 * 1, 2 * 25 * 18, {1, 25, 1 }, {2, 25, 18 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 63 * 1, 2 * 63 * 34, {1, 63, 1 }, {2, 63, 34 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 63 * 1, 2 * 63 * 65, {1, 63, 1 }, {2, 63, 65 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 64 * 1, 2 * 64 * 128, {1, 64, 1 }, {2, 64, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 65 * 1, 2 * 65 * 128, {1, 65, 1 }, {2, 65, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 12 * 1, 2 * 12 * 128, {1, 12, 1 }, {2, 12, 128 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 12 * 1, 2 * 12 * 129, {1, 12, 1 }, {2, 12, 129 } },
    BroadcastTestParams<3>{ MainKernelBroadcast<float, 3>, 1 * 12 * 128, 2 * 12 * 128, {1, 12, 128 }, {2, 12, 128 } }));

TEST_P(BroadcastTestsuite_dim3, BroadcastTestCase)
{
    auto param = GetParam();
    auto &srcShape = param.srcShape;
    auto &dstShape = param.dstShape;
    uint8_t srcGm[param.srcSize] = {0};
    uint8_t dstGm[param.dstSize] = {0};

    param.cal_func(dstGm, srcGm, srcShape, dstShape);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

using BroadcastTestsuite_dim4 = BroadcastTestsuite<4>;

INSTANTIATE_TEST_CASE_P(TEST_Broadcast, BroadcastTestsuite_dim4,
    ::testing::Values(BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 2, 2 * 2 * 6 * 2, {1, 2, 1, 2 }, {1, 2, 6, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 5, 2 * 2 * 20 * 5, {1, 2, 1, 5 }, {2, 2, 20, 5 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 5, 2 * 2 * 28 * 5, {1, 2, 1, 5 }, {2, 2, 28, 5 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 25, 2 * 2 * 28 * 25, {1, 2, 1, 25 }, {2, 2, 28, 25 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 63, 2 * 2 * 128 * 63, {1, 2, 1, 63 }, {2, 2, 128, 63 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 128, 2 * 2 * 128 * 128, {1, 2, 1, 128 }, {2, 2, 128, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 129, 2 * 2 * 128 * 129, {1, 2, 1, 129 }, {2, 2, 128, 129 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 256, 2 * 2 * 128 * 256, {1, 2, 1, 256 }, {2, 2, 128, 256 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 1 * 257, 2 * 2 * 128 * 257, {1, 2, 1, 257 }, {2, 2, 128, 257 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 1 * 2 * 128 * 129, 2 * 2 * 128 * 129, {1, 2, 128, 129 }, {2, 2, 128, 129 } },

    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 29, {2, 1, 2, 1 }, {2, 2, 2, 29 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 4 * 1, 2 * 2 * 4 * 29, {2, 1, 4, 1 }, {2, 2, 4, 29 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 5 * 1, 2 * 2 * 5 * 29, {2, 1, 5, 1 }, {2, 2, 5, 29 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 10 * 1, 2 * 2 * 10 * 29, {2, 1, 10, 1 }, {2, 2, 10, 29 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 3 * 1, 2 * 2 * 3 * 127, {2, 1, 3, 1 }, {2, 2, 3, 127 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 63 * 1, 2 * 2 * 63 * 128, {2, 1, 63, 1 }, {2, 2, 63, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 129, {2, 1, 65, 1 }, {2, 2, 65, 129 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 257, {2, 1, 65, 1 }, {2, 2, 65, 257 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 256, {2, 1, 65, 1 }, {2, 2, 65, 256 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 288, {2, 1, 65, 1 }, {2, 2, 65, 288 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 128 * 1, 2 * 2 * 128 * 128, {2, 1, 128, 1 }, {2, 2, 128, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 129 * 1, 2 * 2 * 129 * 128, {2, 1, 129, 1 }, {2, 2, 129, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<uint8_t, 4>, 2 * 1 * 129 * 128, 2 * 2 * 129 * 128, {2, 1, 129, 128 }, {2, 2, 129, 128 } },

    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 2, 2 * 2 * 15 * 2, {1, 2, 1, 2 }, {2, 2, 15, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 2, 2 * 2 * 21 * 2, {1, 2, 1, 2 }, {2, 2, 21, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 23 * 1 * 2, 2 * 23 * 21 * 2, {1, 23, 1, 2 }, {2, 23, 21, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 23 * 1 * 21, 2 * 23 * 7 * 21, {1, 23, 1, 21 }, {2, 23, 7, 21 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 63, 2 * 2 *128 * 63, {1, 2, 1, 63 }, {2, 2, 128, 63 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 64, 2 * 2 *128 * 64, {1, 2, 1, 64 }, {2, 2, 128, 64 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 65, 2 * 2 *128 * 65, {1, 2, 1, 65 }, {2, 2, 128, 65 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 128, 2 * 2 * 128 * 128, {1, 2, 1, 128 }, {2, 2, 128, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 144, 2 * 2 * 128 * 144, {1, 2, 1, 144 }, {2, 2, 128, 144 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 1 * 129, 2 * 2 * 128 * 129, {1, 2, 1, 129 }, {2, 2, 128, 129 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 1 * 2 * 128 * 129, 2 * 2 * 128 * 129, {1, 2, 128, 129 }, {2, 2, 128, 129 } },

    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 15, {2, 1, 2, 1 }, {2, 2, 2, 15 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 31, {2, 1, 2, 1 }, {2, 2, 2, 31 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 5 * 1, 2 * 2 * 5 * 21, {2, 1, 5, 1 }, {2, 2, 5, 21 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 21, {2, 1, 9, 1 }, {2, 2, 9, 21 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 65, {2, 1, 2, 1 }, {2, 2, 2, 65 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 63 * 1, 2 * 2 * 63 * 128, {2, 1, 63, 1 }, {2, 2, 63, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 64 * 1, 2 * 2 * 64 * 128, {2, 1, 64, 1 }, {2, 2, 64, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 128, {2, 1, 65, 1 }, {2, 2, 65, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 128 * 1, 2 * 2 * 128 * 128, {2, 1, 128, 1 }, {2, 2, 128, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 129 * 1, 2 * 2 * 129 * 128, {2, 1, 129, 1 }, {2, 2, 129, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<half, 4>, 2 * 1 * 129 * 128, 2 * 2 * 129 * 128, {2, 1, 129, 128 }, {2, 2, 129, 128 } },

    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 2, 2 * 2 * 7 * 2, {1, 2, 1, 2 }, {2, 2, 7, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 2, 2 * 2 * 23 * 2, {1, 2, 1, 2 }, {2, 2, 23, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 2, 2 * 2 * 31 * 2, {1, 2, 1, 2 }, {2, 2, 31, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 2, 2 * 2 * 128 * 2, {1, 2, 1, 2 }, {2, 2, 128, 2 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 8, 2 * 2 * 3 * 8, {1, 2, 1, 8 }, {2, 2, 3, 8 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 8, 2 * 2 * 5 * 8, {1, 2, 1, 8 }, {2, 2, 5, 8 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 8, 2 * 2 * 25 * 8, {1, 2, 1, 8 }, {2, 2, 25, 8 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 16, 2 * 2 * 2 * 16, {1, 2, 1, 16 }, {2, 2, 2, 16 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 16, 2 * 2 * 25 * 16, {1, 2, 1, 16 }, {2, 2, 25, 16 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 24, 2 * 2 * 2 * 24, {1, 2, 1, 24 }, {2, 2, 2, 24 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 24, 2 * 2 * 25 * 24, {1, 2, 1, 24 }, {2, 2, 25, 24 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 32, 2 * 2 * 12 * 32, {1, 2, 1, 32 }, {2, 2, 12, 32 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 63, 2 * 2 * 128 * 63, {1, 2, 1, 63 }, {2, 2, 128, 63 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 64, 2 * 2 * 128 * 64, {1, 2, 1, 64 }, {2, 2, 128, 64 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 65, 2 * 2 * 128 * 65, {1, 2, 1, 65 }, {2, 2, 128, 65 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 128, 2 * 2 * 12 * 128, {1, 2, 1, 128 }, {2, 2, 12, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 1 * 129, 2 * 2 * 12 * 129, {1, 2, 1, 129 }, {2, 2, 12, 129 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 1 * 2 * 12 * 129, 2 * 2 * 12 * 129, {1, 2, 12, 129 }, {2, 2, 12, 129 } },

    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 8 * 1, 2 * 2 * 8 * 8, {2, 1, 8, 1 }, {2, 2, 8, 8 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 2 * 1, 2 * 2 * 2 * 12, {2, 1, 2, 1 }, {2, 2, 2, 12 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 4 * 1, 2 * 2 * 4 * 12, {2, 1, 4, 1 }, {2, 2, 4, 12 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 25 * 1, 2 * 2 * 25 * 18, {2, 1, 25, 1 }, {2, 2, 25, 18 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 63 * 1, 2 * 2 * 63 * 34, {2, 1, 63, 1 }, {2, 2, 63, 34 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 63 * 1, 2 * 2 * 63 * 65, {2, 1, 63, 1 }, {2, 2, 63, 65 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 64 * 1, 2 * 2 * 64 * 128, {2, 1, 64, 1 }, {2, 2, 64, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 65 * 1, 2 * 2 * 65 * 128, {2, 1, 65, 1 }, {2, 2, 65, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 12 * 1, 2 * 2 * 12 * 128, {2, 1, 12, 1 }, {2, 2, 12, 128 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 12 * 1, 2 * 2 * 12 * 128, {2, 1, 12, 1 }, {2, 2, 12, 129 } },
    BroadcastTestParams<4>{ MainKernelBroadcast<float, 4>, 2 * 1 * 12 * 128, 2 * 2 * 12 * 128, {2, 1, 12, 128 }, {2, 2, 12, 128 } }));

TEST_P(BroadcastTestsuite_dim4, BroadcastTestCase)
{
    auto param = GetParam();
    auto &srcShape = param.srcShape;
    auto &dstShape = param.dstShape;
    uint8_t srcGm[param.srcSize] = {0};
    uint8_t dstGm[param.dstSize] = {0};

    param.cal_func(dstGm, srcGm, srcShape, dstShape);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

using BroadcastTestsuite_dim9 = BroadcastTestsuite<9>;

INSTANTIATE_TEST_CASE_P(TEST_Broadcast, BroadcastTestsuite_dim9,
    ::testing::Values(
    BroadcastTestParams<9>{ MainKernelBroadcast<uint8_t, 9>, 2 * 1 * 2 * 1 * 2 * 1 * 2 * 1 * 2,
                                                             2 * 2 * 2 * 2 * 2 * 2 * 2 * 6 * 2,
                                                            {2, 1, 2, 1, 2, 1, 2, 1, 2 }, {2, 2, 2, 2, 2, 2, 2, 6, 2 } },

    BroadcastTestParams<9>{ MainKernelBroadcast<uint8_t, 9>, 1 * 2 * 1 * 2 * 1 * 2 * 1 * 2 * 1,
                                                             2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 29,
                                                            {1, 2, 1, 2, 1, 2, 1, 2, 1 }, {2, 2, 2, 2, 2, 2, 2, 2, 29 } },

    BroadcastTestParams<9>{ MainKernelBroadcast<half, 9>, 2 * 1 * 2 * 1 * 2 * 1 * 2 * 1 * 2,
                                                          2 * 2 * 2 * 2 * 2 * 2 * 2 * 15 * 2,
                                                         {2, 1, 2, 1, 2, 1, 2, 1, 2 }, {2, 2, 2, 2, 2, 2, 2, 15, 2 } },

    BroadcastTestParams<9>{ MainKernelBroadcast<half, 9>, 1 * 2 * 1 * 2 * 1 * 2 * 1 * 2 * 1,
                                                          2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 15,
                                                         {1, 2, 1, 2, 1, 2, 1, 2, 1 }, {2, 2, 2, 2, 2, 2, 2, 2, 15 } },

    BroadcastTestParams<9>{ MainKernelBroadcast<float, 9>, 2 * 1 * 2 * 1 * 2 * 1 * 2 * 1 * 2,
                                                           2 * 2 * 2 * 2 * 2 * 2 * 2 * 7 * 2,
                                                          {2, 1, 2, 1, 2, 1, 2, 1, 2 }, {2, 2, 2, 2, 2, 2, 2, 7, 2 } },

    BroadcastTestParams<9>{ MainKernelBroadcast<float, 9>, 1 * 2 * 1 * 2 * 1 * 2 * 1 * 8 * 1,
                                                           2 * 2 * 2 * 2 * 2 * 2 * 2 * 8 * 8,
                                                          {1, 2, 1, 2, 1, 2, 1, 8, 1 }, {2, 2, 2, 2, 2, 2, 2, 8, 8 } }));

TEST_P(BroadcastTestsuite_dim9, BroadcastTestCase)
{
    auto param = GetParam();
    auto &srcShape = param.srcShape;
    auto &dstShape = param.dstShape;
    uint8_t srcGm[param.srcSize] = {0};
    uint8_t dstGm[param.dstSize] = {0};

    param.cal_func(dstGm, srcGm, srcShape, dstShape);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
