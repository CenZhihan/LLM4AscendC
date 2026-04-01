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
#define private public
#define protected public
#include "kernel_operator.h"
#include "kernel_utils.h"
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AscendC;

namespace AscendC {
template <typename srcType, typename dstType, typename scaleType, bool has_offset, int kdim, bool has_workspace>
class KernelAscendDeQuantPerGroup {
public:
    __aicore__ inline KernelAscendDeQuantPerGroup() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR scale_gm, GM_ADDR offset_gm,
                                uint32_t m_num, uint32_t n_num, uint32_t calCount_num, uint32_t gs_num)
    {
        m = m_num;
        n = n_num;
        calCount = calCount_num;
        gs = gs_num;
        out_m = calCount_num / n;
        scaleK = ((n_num + gs_num - 1) / gs_num);
        uint32_t align_size = 1024;
        if constexpr (kdim == 0) {
            scaleK = ((m_num + gs_num - 1) / gs_num);
            align_size = scaleK * n_num;
        }
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), m*n);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ dstType *>(dst_gm), m*n);
        scale_global.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType *>(scale_gm), align_size);
        offset_global.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType *>(offset_gm), align_size);

        pipe.InitBuffer(inQueue, 1, m * n * sizeof(srcType));
        pipe.InitBuffer(inQueueScale, 1, align_size * sizeof(scaleType));
        pipe.InitBuffer(inQueueOffset, 1, align_size * sizeof(scaleType));
        pipe.InitBuffer(outQueue, 1, m * n * sizeof(dstType));
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
        LocalTensor<srcType> srcLocal = inQueue.AllocTensor<srcType>();
        LocalTensor<scaleType> scaleLocal = inQueueScale.AllocTensor<scaleType>();
        LocalTensor<scaleType> offsetLocal = inQueueOffset.AllocTensor<scaleType>();

        DataCopy(srcLocal, src_global, m*n);
        if constexpr (kdim == 0) {
            int align_size = (n*scaleK * sizeof(scaleType) + 31) / 32 * 32 / sizeof(scaleType);
            DataCopy(scaleLocal, scale_global, align_size);
            DataCopy(offsetLocal, offset_global, align_size);
        } else {
            int align_size = (m*scaleK * sizeof(scaleType) + 31) / 32 * 32 / sizeof(scaleType);
            DataCopy(scaleLocal, scale_global, align_size);
            DataCopy(offsetLocal, offset_global, align_size);
        }
        inQueue.EnQue(srcLocal);
        inQueueScale.EnQue(scaleLocal);
        inQueueOffset.EnQue(offsetLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<dstType> dstLocal = outQueue.AllocTensor<dstType>();

        LocalTensor<srcType> srcLocal = inQueue.DeQue<srcType>();
        LocalTensor<scaleType> scaleLocal = inQueueScale.DeQue<scaleType>();
        LocalTensor<scaleType> offsetLocal = inQueueOffset.DeQue<scaleType>();

        constexpr static bool isReuseSource = false;
        constexpr static AscendDeQuantConfig config = {has_offset, kdim};
        constexpr static AscendDeQuantPolicy policy = AscendDeQuantPolicy::PER_GROUP;
        AscendDeQuantParam para;
        para.m = m;
        para.n = n;
        para.calCount = calCount;
        para.groupSize = gs;
        if constexpr(has_workspace) {
            LocalTensor<uint8_t> sharedTmpBuffer;
            bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
            AscendDequant<dstType, srcType, scaleType, config, policy>(dstLocal, srcLocal, scaleLocal, offsetLocal, sharedTmpBuffer, para);
        } else {
            AscendDequant<dstType, srcType, scaleType, config, policy>(dstLocal, srcLocal, scaleLocal, offsetLocal, para);
        }
        outQueue.EnQue<dstType>(dstLocal);
        inQueue.FreeTensor(srcLocal);
        inQueueScale.FreeTensor(scaleLocal);
        inQueueOffset.FreeTensor(offsetLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dstType> dstLocal = outQueue.DeQue<dstType>();
        DataCopy(dst_global, dstLocal, out_m * n);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<srcType> src_global;
    GlobalTensor<scaleType> scale_global;
    GlobalTensor<scaleType> offset_global;
    GlobalTensor<dstType> dst_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue, inQueueScale, inQueueOffset;
    TQue<TPosition::VECOUT, 1> outQueue;

    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    uint32_t out_m;
    uint32_t gs;
    int32_t scaleK;
};
} // namespace AscendC

template <typename srcType, typename dstType, typename scaleType, bool has_offset, int kdim, bool has_workspace>
__global__ __aicore__ void Entry(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* scaleGm, __gm__ uint8_t* offsetGm,
                                 uint32_t m, uint32_t n, uint32_t calCount, uint32_t gs)
{
    AscendC::KernelAscendDeQuantPerGroup<srcType, dstType, scaleType, has_offset, kdim, has_workspace> op;
    op.Init(srcGm, dstGm, scaleGm, offsetGm, m, n, calCount, gs);
    op.Process();
}

struct AscendDeQuantPerGroupTestParams {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    uint32_t gs;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
};

class AscendDeQuantPerGroupTestsuite : public testing::Test, public testing::WithParamInterface<AscendDeQuantPerGroupTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ASCEND_QUANT_PER_GROUP, AscendDeQuantPerGroupTestsuite,
    ::testing::Values(
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<int32_t, half, half, false, 1, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 64, Entry<int32_t, bfloat16_t, bfloat16_t, false, 1, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 128, Entry<int32_t, float, float, false, 1, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 256, Entry<int32_t, float, half, false, 1, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<int32_t, float, bfloat16_t, false, 1, true> },
        
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<float, half, half, false, 1, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 64, Entry<float, bfloat16_t, bfloat16_t, false, 1, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 128, Entry<float, float, float, false, 1, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 256, Entry<float, float, half, false, 1, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<float, float, bfloat16_t, false, 1, true> },

        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<int32_t, half, half, false, 0, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 64, Entry<int32_t, bfloat16_t, bfloat16_t, false, 0, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 128, Entry<int32_t, float, float, false, 0, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 256, Entry<int32_t, float, half, false, 0, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<int32_t, float, bfloat16_t, false, 0, true> },
        
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<float, half, half, false, 0, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 64, Entry<float, bfloat16_t, bfloat16_t, false, 0, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 128, Entry<float, float, float, false, 0, true> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 256, Entry<float, float, half, false, 0, false> },
        AscendDeQuantPerGroupTestParams{ 10, 256, 1280, 32, Entry<float, float, bfloat16_t, false, 0, true> }
    ));

TEST_P(AscendDeQuantPerGroupTestsuite, AscendDeQuantPerGroupOpTestCase)
{
    const int dataBytes = 4;
    auto param = GetParam();
    uint8_t srcGm[param.m * param.n * dataBytes];
    uint8_t dstGm[param.m * param.n * dataBytes];
    uint8_t scaleGm[param.m * dataBytes];
    uint8_t offsetGm[param.m * dataBytes];
    param.cal_func(srcGm, dstGm, scaleGm, offsetGm, param.m, param.n, param.calCount, param.gs);
}
