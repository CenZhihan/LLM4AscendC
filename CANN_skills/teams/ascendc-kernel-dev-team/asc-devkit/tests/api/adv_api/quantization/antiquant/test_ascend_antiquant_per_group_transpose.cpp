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
template <typename srcType, typename dstType, typename scaleType, bool has_offset, bool has_transpose, int kdim>
class KernelAscendAntiQuantPerGroupTranspose {
public:
    __aicore__ inline KernelAscendAntiQuantPerGroupTranspose() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR scale_gm, GM_ADDR offset_gm,
                                uint32_t m_num, uint32_t n_num, uint32_t calCount_num, uint32_t gs_num)
    {
        m = m_num;
        n = n_num;
        calCount = calCount_num;
        out_m = calCount / n;
        gs = gs_num;
        int align_size = 0;
        if constexpr (kdim == 0) {
            scaleK = ((n_num + gs_num - 1) / gs_num);
            align_size = scaleK * m_num;
        } else {
            scaleK = ((m_num + gs_num - 1) / gs_num);
            align_size = scaleK * n_num;
        }

        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), m*n);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ dstType *>(dst_gm), out_m*n);
        scale_global.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType *>(scale_gm), align_size);
        offset_global.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType *>(offset_gm), align_size);

        pipe.InitBuffer(inQueue, 1, m * n * sizeof(srcType));
        pipe.InitBuffer(inQueueScale, 1, align_size * sizeof(scaleType));
        pipe.InitBuffer(inQueueOffset, 1, align_size * sizeof(scaleType));
        pipe.InitBuffer(outQueue, 1, out_m * n * sizeof(dstType));
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
            int align_size = (m*scaleK * sizeof(dstType) + 31) / 32 * 32 / sizeof(dstType);
            DataCopy(scaleLocal, scale_global, align_size);
            DataCopy(offsetLocal, offset_global, align_size);
        } else {
            int align_size = (n*scaleK * sizeof(dstType) + 31) / 32 * 32 / sizeof(dstType);
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
        constexpr static AscendAntiQuantConfig config = {has_offset, has_transpose, kdim};
        constexpr static AscendAntiQuantPolicy policy = AscendAntiQuantPolicy::PER_GroupTranspose;
        AscendAntiQuantParam para;
        para.m = m;
        para.n = n;
        para.calCount = calCount;
        para.GroupTransposeSize = gs;

        AscendAntiQuant<dstType, srcType, scaleType, config, policy>(dstLocal, srcLocal, scaleLocal, offsetLocal, para);

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
    uint32_t scaleK;
};
} // namespace AscendC

template <typename srcType, typename dstType, typename scaleType, bool has_offset, bool has_transpose, int kdim>
__global__ __aicore__ void Entry(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* scaleGm, __gm__ uint8_t* offsetGm,
                                 uint32_t m, uint32_t n, uint32_t calCount, uint32_t gs)
{
    AscendC::KernelAscendAntiQuantPerGroupTranspose<srcType, dstType, scaleType, has_offset, has_transpose, kdim> op;
    op.Init(srcGm, dstGm, scaleGm, offsetGm, m, n, calCount, gs);
    op.Process();
}

struct AscendAntiQuantPerGroupTransposeTestParams {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    int gs;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
};

class AscendAntiQuantPerGroupTransposeTestsuite : public testing::Test, public testing::WithParamInterface<AscendAntiQuantPerGroupTransposeTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ASCEND_ANTIQUANT_PER_GroupTranspose, AscendAntiQuantPerGroupTransposeTestsuite,
    ::testing::Values(
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<int8_t, half, half, true, true, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<int8_t, bfloat16_t, bfloat16_t, true, false, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<int8_t, float, float, true, false, 1> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<hifloat8_t, half, half, true, true, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<hifloat8_t, bfloat16_t, bfloat16_t, true, false, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<hifloat8_t, float, float, true, false, 1> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<fp8_e5m2_t, half, half, true, true, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<fp8_e5m2_t, bfloat16_t, bfloat16_t, true, false, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<fp8_e5m2_t, float, float, true, false, 1> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<fp8_e4m3fn_t, half, half, true, true, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<fp8_e4m3fn_t, bfloat16_t, bfloat16_t, true, false, 1> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<fp8_e4m3fn_t, float, float, true, false, 1> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<int8_t, half, half, true, true, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<int8_t, bfloat16_t, bfloat16_t, true, false, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<int8_t, float, float, true, false, 0> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<hifloat8_t, half, half, true, true, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<hifloat8_t, bfloat16_t, bfloat16_t, true, false, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<hifloat8_t, float, float, true, false, 0> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<fp8_e5m2_t, half, half, true, true, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<fp8_e5m2_t, bfloat16_t, bfloat16_t, true, false, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<fp8_e5m2_t, float, float, true, false, 0> },

        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 32, Entry<fp8_e4m3fn_t, half, half, true, true, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 65, 128, 8320, 64, Entry<fp8_e4m3fn_t, bfloat16_t, bfloat16_t, true, false, 0> },
        AscendAntiQuantPerGroupTransposeTestParams{ 130, 128, 16640, 128, Entry<fp8_e4m3fn_t, float, float, true, false, 0> }
    ));

TEST_P(AscendAntiQuantPerGroupTransposeTestsuite, AscendAntiQuantPerGroupTransposeOpTestCase)
{
    const int dataBytes = 4;
    auto param = GetParam();
    uint8_t srcGm[param.m * param.n * dataBytes];
    uint8_t dstGm[param.m * param.n * dataBytes];
    uint8_t scaleGm[param.m * dataBytes];
    uint8_t offsetGm[param.m * dataBytes];
    param.cal_func(srcGm, dstGm, scaleGm, offsetGm, param.m, param.n, param.calCount, param.gs);
}
