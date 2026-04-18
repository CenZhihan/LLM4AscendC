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
class KernelMatmulSp {
public:
    __aicore__ inline KernelMatmulSp() {}
    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* idx, __gm__ uint8_t* c, uint16_t m, uint16_t k, uint16_t n)
    {
        this->m = m;
        this->k = k;
        this->n = n;

        aSize = m * k;
        bSize = k / 2 * n;
        cSize = m * n;
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 32;

        aGM.SetGlobalBuffer((__gm__ int8_t*)a);
        bGM.SetGlobalBuffer((__gm__ int8_t*)b);
        idxGM.SetGlobalBuffer((__gm__ uint8_t*)idx);
        cGM.SetGlobalBuffer((__gm__ int32_t*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(int8_t));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(int8_t));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(int8_t));
        pipe.InitBuffer(inQueueIdxB1, 1, (bSize / 4) * sizeof(int8_t));
        pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(int8_t));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        LocalTensor<TensorTrait<int8_t>> b1Local = inQueueB1.DeQue<TensorTrait<int8_t>>();
        LocalTensor<TensorTrait<uint8_t>> idexb1Local = inQueueIdxB1.DeQue<TensorTrait<uint8_t>>();
        LocalTensor<TensorTrait<int8_t>> a2Local = inQueueA2.DeQue<TensorTrait<int8_t>>();
        SplitB(b1Local, idexb1Local);
        Compute(a2Local);
        inQueueB1.FreeTensor(b1Local);
        inQueueIdxB1.FreeTensor(idexb1Local);
        inQueueA2.FreeTensor(a2Local);

        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<TensorTrait<int8_t>> a1Local = inQueueA1.AllocTensor<TensorTrait<int8_t>>();
        LocalTensor<TensorTrait<int8_t>> b1Local = inQueueB1.AllocTensor<TensorTrait<int8_t>>();
        LocalTensor<TensorTrait<uint8_t>> idxb1Local = inQueueIdxB1.AllocTensor<TensorTrait<uint8_t>>();
        DataCopy(a1Local, aGM, { 1, static_cast<uint16_t>(aSize * sizeof(int8_t) / 32), 0, 0 });
        DataCopy(b1Local, bGM, { 1, static_cast<uint16_t>(bSize * sizeof(int8_t) / 32), 0, 0 });
        DataCopy(idxb1Local, idxGM, { 1, static_cast<uint16_t>(bSize / 4 * sizeof(int8_t) / 32), 0, 0 });

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueIdxB1.EnQue(idxb1Local);
    }
    __aicore__ inline void SplitA()
    {
        int srcOffset = 0;
        int dstOffset = 0;
        LocalTensor<TensorTrait<int8_t>> a1Local = inQueueA1.DeQue<TensorTrait<int8_t>>();
        LocalTensor<TensorTrait<int8_t>> a2Local = inQueueA2.AllocTensor<TensorTrait<int8_t>>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks * mBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;

        LoadData(a2Local, a1Local, loadDataParams);

        inQueueA2.EnQue<TensorTrait<int8_t>>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(LocalTensor<TensorTrait<int8_t>>& b1Local, LocalTensor<TensorTrait<uint8_t>>& idxb1Local)
    {
        LocalTensor<TensorTrait<int8_t>> b2Local = inQueueB2.AllocTensor<TensorTrait<int8_t>>();

        // transform nz to zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks * nBlocks / 2;
        loadDataParams.srcStride = 0;
        loadDataParams.ifTranspose = false;

        LoadDataWithSparse(b2Local, b1Local, idxb1Local, loadDataParams);

        inQueueB2.EnQue<TensorTrait<int8_t>>(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<TensorTrait<int8_t>>& a2Local)
    {
        LocalTensor<TensorTrait<int8_t>> b2Local = inQueueB2.DeQue<TensorTrait<int8_t>>();
        LocalTensor<TensorTrait<int32_t>> c1Local = outQueueCO1.AllocTensor<TensorTrait<int32_t>>();

        MmadWithSparse(c1Local, a2Local, b2Local, { m, n, k, false, 0, false, false, false });

        outQueueCO1.EnQue<TensorTrait<int32_t>>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<TensorTrait<int32_t>> c1Local = outQueueCO1.DeQue<TensorTrait<int32_t>>();

        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;

        Fixpipe(cGM, c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    TPipe pipe;

    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B1, 1> inQueueIdxB1;
    TQue<TPosition::B2, 1> inQueueB2;
    // dst queue
    TQue<TPosition::CO1, 1> outQueueCO1;

    GlobalTensor<TensorTrait<int8_t>> aGM, bGM;
    GlobalTensor<TensorTrait<uint8_t>> idxGM;
    GlobalTensor<TensorTrait<int32_t>> cGM;

    uint16_t m;
    uint16_t n;
    uint16_t k;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};
}

class TEST_MMAD_SP : public testing::Test {
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

TEST_F(TEST_MMAD_SP, MMAD_SP_Case)
{
    uint16_t m = 16;
    uint16_t n = 16;
    uint16_t k = 64;

    if ASCEND_IS_AIV {
        return;
    }
    uint8_t a[m * k * sizeof(int8_t)] = {0};
    uint8_t b[n * k * sizeof(int8_t) / 2] = {0};
    uint8_t idx[m * k * sizeof(int32_t) / 2 / 4] = {0};
    uint8_t c[m * n * sizeof(int32_t)] = {0};
    AscendC::KernelMatmulSp op;
    op.Init(a, b, idx, c, m, k, n);
    op.Process();
    
    for (int32_t i = 0; i < m * n * sizeof(int32_t); i++) {
        EXPECT_EQ(c[i], 0x00);
    }
}