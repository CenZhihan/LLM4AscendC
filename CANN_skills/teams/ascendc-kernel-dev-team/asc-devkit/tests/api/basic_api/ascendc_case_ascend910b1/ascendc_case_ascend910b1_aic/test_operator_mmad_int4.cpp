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
#include "test_utils.h"

using namespace std;


namespace AscendC {
class KernelMatmulInt4 {
public:
    __aicore__ inline KernelMatmulInt4()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 64;
    }
    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, bool is2d)
    {
        aGM.SetGlobalBuffer((__gm__ int4b_t*)a);
        bGM.SetGlobalBuffer((__gm__ int4b_t*)b);
        cGM.SetGlobalBuffer((__gm__ int32_t*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(int4b_t));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(int4b_t));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(int4b_t));
        pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(int4b_t));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(int32_t));
        this->is2d = is2d;
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        LocalTensor<TensorTrait<int4b_t>> b1Local = inQueueB1.DeQue<TensorTrait<int4b_t>>();
        LocalTensor<TensorTrait<int4b_t>> a2Local = inQueueA2.DeQue<TensorTrait<int4b_t>>();
        SplitB(b1Local);
        Compute(a2Local);
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);

        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<TensorTrait<int4b_t>> a1Local = inQueueA1.AllocTensor<TensorTrait<int4b_t>>();
        LocalTensor<TensorTrait<int4b_t>> b1Local = inQueueB1.AllocTensor<TensorTrait<int4b_t>>();

        DataCopy(a1Local, aGM, { 1, static_cast<uint16_t>(16*64 * sizeof(int4b_t) / 2 / 32), 0, 0 });
        DataCopy(b1Local, bGM, { 1, static_cast<uint16_t>(64*64 * sizeof(int4b_t) / 2 / 32), 0, 0 });

        LocalTensor<float> ub1(TPosition::VECIN, 0, 256);
        LocalTensor<float> ub2(TPosition::VECIN, 256, 256);
        // Test when AIC, npu debug will not check mask[] or mask
        BinaryRepeatParams params;
        uint64_t mask[2] = {3, 3};
        Add(ub2, ub1, ub2, 30, 1, params);
        Add(ub2, ub1, ub2, mask, 1, params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        int srcOffset = 0;
        int dstOffset = 0;
        LocalTensor<TensorTrait<int4b_t>> a1Local = inQueueA1.DeQue<TensorTrait<int4b_t>>();
        LocalTensor<TensorTrait<int4b_t>> a2Local = inQueueA2.AllocTensor<TensorTrait<int4b_t>>();

        if (is2d) {
            // load2d
            LoadData2DParams loadDataParams;
            loadDataParams.repeatTimes = kBlocks * mBlocks;
            loadDataParams.srcStride = 0;
            loadDataParams.ifTranspose = false;

            LoadData(a2Local, a1Local, loadDataParams);

            // cover
            auto a1TmpBuf = a1Local.ReinterpretCast<TensorTrait<int8_t>>();
            auto a2TmpBuf = a2Local.ReinterpretCast<TensorTrait<int8_t>>();
            LoadData(a2TmpBuf, a1TmpBuf, loadDataParams);
            loadDataParams.ifTranspose = true;
            LoadData(a2TmpBuf, a1TmpBuf, loadDataParams);

        } else {
            //load3d
            LoadData3DParamsV2<int4b_t> loadData3dParams;
            loadData3dParams.l1W = 8;
            loadData3dParams.l1H = 2;
            loadData3dParams.channelSize = 64;
            loadData3dParams.kExtension = 64;
            loadData3dParams.mExtension = 16;
            loadData3dParams.kStartPt = 0;
            loadData3dParams.mStartPt = 0;
            loadData3dParams.strideW = 1;
            loadData3dParams.strideH = 1;
            loadData3dParams.filterW = 1;
            loadData3dParams.filterH = 1;
            loadData3dParams.dilationFilterW = 1;
            loadData3dParams.dilationFilterH = 1;
            loadData3dParams.enTranspose = false;
            loadData3dParams.enSmallK = false;
            loadData3dParams.padValue = 0;

            LoadData(a2Local, a1Local, loadData3dParams);
        }

        inQueueA2.EnQue(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(LocalTensor<TensorTrait<int4b_t>>& b1Local)
    {
        LocalTensor<TensorTrait<int4b_t>> b2Local = inQueueB2.AllocTensor<TensorTrait<int4b_t>>();

        // transform nz to zn
        if (is2d) {
            // load2d
            LoadData2DParams loadDataParams;
            loadDataParams.repeatTimes = 1;
            loadDataParams.srcStride = 0;
            loadDataParams.ifTranspose = false;

            LoadData(b2Local, b1Local, loadDataParams);
            //cover
            auto b1TmpBuf = b1Local.ReinterpretCast<TensorTrait<int8_t>>();
            auto b2TmpBuf = b2Local.ReinterpretCast<TensorTrait<int8_t>>();
            loadDataParams.ifTranspose = true;
            LoadData(b2TmpBuf, b1TmpBuf, loadDataParams);

        } else {
            //load2dwithtranspose
            LoadData2dTransposeParams loadDataParams;
            loadDataParams.startIndex = 0;
            loadDataParams.srcStride = 1;
            loadDataParams.addrMode = 0;

            loadDataParams.repeatTimes = 1;
            loadDataParams.dstGap = 16;
            loadDataParams.dstFracGap = 0;
            LoadDataWithTranspose<TensorTrait<int4b_t>>(b2Local, b1Local, loadDataParams);
            //cover
            auto b1TmpBuf = b1Local.ReinterpretCast<TensorTrait<int8_t>>();
            auto b2TmpBuf = b2Local.ReinterpretCast<TensorTrait<int8_t>>();
            LoadDataWithTranspose<TensorTrait<int8_t>>(b2TmpBuf, b1TmpBuf, loadDataParams);
        }

        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<TensorTrait<int4b_t>>& a2Local)
    {
        LocalTensor<TensorTrait<int4b_t>> b2Local = inQueueB2.DeQue<TensorTrait<int4b_t>>();
        LocalTensor<TensorTrait<int32_t>> c1Local = outQueueCO1.AllocTensor<TensorTrait<int32_t>>();

        Mmad(c1Local, a2Local, b2Local, { m, n, k, false, 0, false, false, false });

        outQueueCO1.EnQue<TensorTrait<int32_t>>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<TensorTrait<int32_t>> c1Local = outQueueCO1.DeQue<TensorTrait<int32_t>>();

        FixpipeParams<int32_t> fixpipeParams;
        fixpipeParams.cburstNum = n / 16;
        fixpipeParams.burstLen = m * 16 * sizeof(int32_t) / 32;
        fixpipeParams.srcStride = 0;
        fixpipeParams.dstStride = 64;

        fixpipeParams.nz2ndParams.nz2ndEn = true;
        fixpipeParams.nz2ndParams.ndNum = 1;
        fixpipeParams.nz2ndParams.srcNdStride = 0;
        fixpipeParams.nz2ndParams.dstNdStride = 0;
        fixpipeParams.nz2ndParams.originalNSize = 64;
        Fixpipe(cGM, c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    TPipe pipe;

    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    // dst queue
    TQue<TPosition::CO1, 1> outQueueCO1;

    GlobalTensor<TensorTrait<int4b_t>> aGM, bGM;
    GlobalTensor<TensorTrait<int32_t>> cGM;

    uint16_t m = 16;
    uint16_t n = 64;
    uint16_t k = 64;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
    bool is2d;
};
}

class TEST_MMAD_INT4 : public testing::Test {
protected:
    void SetUp()
    {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        g_coreType = AscendC::MIX_TYPE;
    }
};

TEST_F(TEST_MMAD_INT4, MMAD_Case_int4_load3dv2)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    GTEST_SKIP() << "basic DataCopy GM -> Local on 2201 does not support int4b_t.";
#endif
    uint16_t m = 16;
    uint16_t n = 64;
    uint16_t k = 64;

    if ASCEND_IS_AIV {
        return;
    }
    uint8_t a[m * k * sizeof(int8_t)] = {0};
    uint8_t b[n * k * sizeof(int8_t)] = {0};
    uint8_t c[m * n * sizeof(int32_t)] = {0};
    AscendC::KernelMatmulInt4 op;
    op.Init(a, b, c, false);
    op.Process();

    for (int32_t i = 0; i < m * n * sizeof(int32_t) / 2; i++) {
        EXPECT_EQ(c[i], 0x00);
    }
}

TEST_F(TEST_MMAD_INT4, MMAD_Case_int4_load2d)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    GTEST_SKIP() << "basic DataCopy GM -> Local on 2201 does not support int4b_t.";
#endif
    uint16_t m = 16;
    uint16_t n = 64;
    uint16_t k = 64;

    if ASCEND_IS_AIV {
        return;
    }
    uint8_t a[m * k * sizeof(int8_t)] = {0};
    uint8_t b[n * k * sizeof(int8_t)] = {0};
    uint8_t c[m * n * sizeof(int32_t)] = {0};
    AscendC::KernelMatmulInt4 op;
    op.Init(a, b, c, true);
    op.Process();

    for (int32_t i = 0; i < m * n * sizeof(int32_t) / 2; i++) {
        EXPECT_EQ(c[i], 0x00);
    }
}
