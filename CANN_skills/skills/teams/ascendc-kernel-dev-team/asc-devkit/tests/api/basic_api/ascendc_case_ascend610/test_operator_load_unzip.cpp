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
using namespace AscendC;
 
class TEST_LOADUNZIP : public testing::Test {
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


namespace AscendC {
template <typename FmapT, typename WeightT, typename DstT> class KernelLoadUnzip {
public:
    __aicore__ inline KernelLoadUnzip(uint32_t dstScopeIn, uint32_t srcLenIn, uint32_t dstLenIn, uint32_t numOfIndexTabEntryIn)
    {
        dstScope = dstScopeIn;
        srcLen = srcLenIn;
        dstLen = dstLenIn;
        numOfIndexTabEntry = numOfIndexTabEntryIn;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ int8_t* fmGm, __gm__ int8_t* weGm, __gm__ int8_t* we1Gm, __gm__ int8_t* index0Gm,
        __gm__ int8_t* index1Gm, __gm__ int8_t* dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ FmapT*)fmGm, dstLen);
        weGlobal.SetGlobalBuffer((__gm__ WeightT*)weGm, dstLen);
        we1Global.SetGlobalBuffer((__gm__ WeightT*)we1Gm, dstLen);
        index0Global.SetGlobalBuffer((__gm__ int8_t*)index0Gm, 512);
        index1Global.SetGlobalBuffer((__gm__ int8_t*)index1Gm, 512);
        dstGlobal.SetGlobalBuffer((__gm__ DstT*)dstGm, dstLen);
        pipe.InitBuffer(inQueueB1, 1, dstLen * sizeof(WeightT));
        pipe.InitBuffer(outQueueUB, 1, dstLen * sizeof(WeightT));

        pipe.InitBuffer(inQueueA1, 1, dstLen * sizeof(FmapT));
        pipe.InitBuffer(inQueueA2, 1, dstLen * sizeof(FmapT));
        pipe.InitBuffer(inQueueB2, 2, dstLen * sizeof(WeightT));
        pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(DstT));
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(DstT));
    }
    __aicore__ inline void Process()
    {

        CopyToB1Unzip();
        CopyToA1();
        CopyToL0A();
        CopyToL0BUnzip();
        Compute();
        Aggregate();
        CopyOut();

    }

private:
    __aicore__ inline void CopyToB1Unzip()
    {
        LocalTensor<WeightT> weightB1 = inQueueB1.AllocTensor<WeightT>();
        LoadUnzipIndex(index0Global, numOfIndexTabEntry);
        LoadDataUnzip(weightB1, weGlobal);
        DataCopy(weightB1, weGlobal, dstLen);
        inQueueB1.EnQue(weightB1);
    }

    __aicore__ inline void CopyToA1()
    {
        LocalTensor<FmapT> a1Local = inQueueA1.AllocTensor<FmapT>();
        DataCopy(a1Local, fmGlobal, dstLen);
        inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void CopyToL0A()
    {
        LocalTensor<FmapT> a1Local = inQueueA1.DeQue<FmapT>();
        LocalTensor<FmapT> a2Local = inQueueA2.AllocTensor<FmapT>();
        LoadData2dParams loadL0AParams;
        loadL0AParams.repeatTimes = 1;
        loadL0AParams.ifTranspose = false;

        LoadData(a2Local, a1Local, loadL0AParams);

        inQueueA2.EnQue<FmapT>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void CopyToL0BUnzip()
    {
        LocalTensor<WeightT> weightB2 = inQueueB2.AllocTensor<WeightT>();
        LoadUnzipIndex(index1Global, numOfIndexTabEntry);
        LoadDataUnzip(weightB2, we1Global);
        inQueueB2.EnQue(weightB2);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<FmapT> a2Local = inQueueA2.DeQue<FmapT>();
        LocalTensor<WeightT> b2Local = inQueueB2.DeQue<WeightT>();
        LocalTensor<DstT> c1Local = outQueueCO1.AllocTensor<DstT>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<DstT>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void Aggregate()
    {
        LocalTensor<DstT> c1Local = outQueueCO1.DeQue<DstT>();
        LocalTensor<DstT> c2Local = outQueueCO2.AllocTensor<DstT>();

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 2;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(c2Local, c1Local, dataCopyParams, enhancedParams);

        outQueueCO2.EnQue(c2Local);
        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> c2Local = outQueueCO2.DeQue<DstT>();
        DataCopy(dstGlobal, c2Local, cSize);
        outQueueCO2.FreeTensor(c2Local);
    }


private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    // weight queue
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 2> inQueueB2;
    // dst queue
    TQue<TPosition::CO1, 2> outQueueCO1;
    TQue<TPosition::CO2, 1> outQueueCO2;
    TQue<TPosition::VECOUT, 1> outQueueUB;
    GlobalTensor<FmapT> fmGlobal;
    GlobalTensor<WeightT> weGlobal, we1Global;
    GlobalTensor<DstT> dstGlobal;
    GlobalTensor<int8_t> index0Global;
    GlobalTensor<int8_t> index1Global;

    uint8_t dstScope;
    uint32_t srcLen, dstLen, numOfIndexTabEntry = 1;
    uint16_t m = 32;
    uint16_t n = 32;
    uint16_t k = 32;
    uint16_t cSize;
};
} // namespace AscendC

#define KERNEL_LOAD_UNZIP(fmapType, weightType, dstType, dstScope, srcLen, dstLen, numOfIndexTabEntry) \
    TEST_F(TEST_LOADUNZIP, LOADUNZIP##_##weightType##_##dstType##_##Case)                                  \
    {                                                                                                        \
        int8_t fmapGm[dstLen * sizeof(fmapType)];                                                          \
        int8_t weightGm[dstLen * sizeof(weightType)];                                                      \
        int8_t weight1Gm[dstLen * sizeof(weightType)];                                                     \
        int8_t index0Gm[8 * sizeof(int8_t)];                                                                 \
        int8_t index1Gm[8 * sizeof(int8_t)];                                                                 \
        int8_t dstGm[dstLen * sizeof(dstType)];                                                            \
        AscendC::KernelLoadUnzip<fmapType, weightType, dstType> op(dstScope, srcLen, dstLen,           \
            numOfIndexTabEntry);                                                                             \
        op.Init(fmapGm, weightGm, weight1Gm, index0Gm, index1Gm, dstGm);                                     \
        op.Process();                                                                                        \
    }
KERNEL_LOAD_UNZIP(int8_t, int8_t, int32_t, 0, 896, 1024, 1);
