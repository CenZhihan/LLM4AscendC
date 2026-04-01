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
template <typename srcType, typename dstType, typename scaleType, bool has_offset, int kdim, bool has_workspace, bool tensor_offset>
class KernelQuantizePerGroup {
public:
    __aicore__ inline KernelQuantizePerGroup() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR scale_gm, GM_ADDR offset_gm,
                                uint32_t m_num, uint32_t n_num, uint32_t calCount_num, uint32_t gs_val)
    {
        m = m_num;
        n = n_num;
        calCount = m * n;
        gs = gs_val;
        scaleK = ((n_num + gs_val - 1) / gs_val);
        uint32_t align_size = 1024;
        if constexpr (kdim == 0) {
            scaleK = ((m_num + gs_val - 1) / gs_val);
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
            int align_size = (n*scaleK * sizeof(dstType) + 31) / 32 * 32 / sizeof(dstType);
            DataCopy(scaleLocal, scale_global, align_size);
            DataCopy(offsetLocal, offset_global, align_size);
        } else {
            int align_size = (m*scaleK * sizeof(dstType) + 31) / 32 * 32 / sizeof(dstType);
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

        constexpr static QuantizeConfig config = {QuantizePolicy::PER_GROUP, has_offset, RoundMode::CAST_RINT, kdim};
        QuantizeParams para;
        para.m = m;
        para.n = n;
        para.groupSize = gs;
        if constexpr (has_workspace) {
            LocalTensor<uint8_t> sharedTmpBuffer;
            bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
            if constexpr (tensor_offset) {
               Quantize<config>(dstLocal, srcLocal, sharedTmpBuffer, scaleLocal, offsetLocal, para);
            } else {
               Quantize<config>(dstLocal, srcLocal, sharedTmpBuffer, scaleLocal, offsetLocal.GetValue(0), para);
            }
        } else {
            if constexpr (tensor_offset) {
               Quantize<config>(dstLocal, srcLocal, scaleLocal, offsetLocal, para);
            } else {
               Quantize<config>(dstLocal, srcLocal, scaleLocal, offsetLocal.GetValue(0), para);
            }
        }
        outQueue.EnQue<dstType>(dstLocal);
        inQueue.FreeTensor(srcLocal);
        inQueueScale.FreeTensor(scaleLocal);
        inQueueOffset.FreeTensor(offsetLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dstType> dstLocal = outQueue.DeQue<dstType>();
        DataCopy(dst_global, dstLocal, m * n);
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
    uint32_t gs;
    uint32_t scaleK;
};
} // namespace AscendC

template <typename srcType, typename dstType, typename scaleType, bool has_offset, int kdim, bool has_workspace, bool tensor_offset>
__global__ __aicore__ void Entry(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* scaleGm, __gm__ uint8_t* offsetGm,
                                 uint32_t m, uint32_t n, uint32_t calCount, uint32_t gs)
{
    AscendC::KernelQuantizePerGroup<srcType, dstType, scaleType, has_offset, kdim, has_workspace, tensor_offset> op;
    op.Init(srcGm, dstGm, scaleGm, offsetGm, m, n, calCount, gs);
    op.Process();
}

struct QuantizePerGroupTestParams {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    int gs;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
};

class QuantizePerGroupTestsuite : public testing::Test, public testing::WithParamInterface<QuantizePerGroupTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ASCEND_QUANT_PER_GROUP, QuantizePerGroupTestsuite,
    ::testing::Values(
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e1m2_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e2m1_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e5m2_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, fp8_e5m2_t, bfloat16_t, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e5m2_t, float, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e5m2_t, float, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e5m2_t, float, true, 1, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e4m3fn_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, fp8_e4m3fn_t, bfloat16_t, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e4m3fn_t, float, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e4m3fn_t, float, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e4m3fn_t, float, true, 1, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, hifloat8_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, hifloat8_t, bfloat16_t, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, hifloat8_t, float, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, hifloat8_t, float, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, hifloat8_t, float, true, 1, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, int8_t, half, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, int8_t, bfloat16_t, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, int8_t, float, true, 1, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, int8_t, float, true, 1, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, int8_t, float, true, 1, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e1m2_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e2m1_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e5m2_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, fp8_e5m2_t, bfloat16_t, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e5m2_t, float, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e5m2_t, float, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e5m2_t, float, true, 0, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e4m3fn_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, fp8_e4m3fn_t, bfloat16_t, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e4m3fn_t, float, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e4m3fn_t, float, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e4m3fn_t, float, true, 0, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, hifloat8_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, hifloat8_t, bfloat16_t, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, hifloat8_t, float, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, hifloat8_t, float, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, hifloat8_t, float, true, 0, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, int8_t, half, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 64, Entry<bfloat16_t, int8_t, bfloat16_t, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, int8_t, float, true, 0, true, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, int8_t, float, true, 0, false, true> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, int8_t, float, true, 0, true, true> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e1m2_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e2m1_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e5m2_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e5m2_t, float, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e5m2_t, float, true, 1, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e5m2_t, float, true, 1, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e4m3fn_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e4m3fn_t, float, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e4m3fn_t, float, true, 1, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e4m3fn_t, float, true, 1, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, hifloat8_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, hifloat8_t, float, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, hifloat8_t, float, true, 1, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, hifloat8_t, float, true, 1, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, int8_t, half, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, int8_t, float, true, 1, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, int8_t, float, true, 1, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, int8_t, float, true, 1, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e1m2_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp4x2_e2m1_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e5m2_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e5m2_t, float, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e5m2_t, float, true, 0, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e5m2_t, float, true, 0, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, fp8_e4m3fn_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, fp8_e4m3fn_t, float, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, fp8_e4m3fn_t, float, true, 0, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, fp8_e4m3fn_t, float, true, 0, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, hifloat8_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, hifloat8_t, float, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, hifloat8_t, float, true, 0, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, hifloat8_t, float, true, 0, true, false> },

        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<half, int8_t, half, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 128, Entry<float, int8_t, float, true, 0, true, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 256, Entry<half, int8_t, float, true, 0, false, false> },
        QuantizePerGroupTestParams{ 10, 256, 1280, 32, Entry<bfloat16_t, int8_t, float, true, 0, true, false> }
    ));

TEST_P(QuantizePerGroupTestsuite, QuantizePerGroupOpTestCase)
{
    const int dataBytes = 4;
    auto param = GetParam();
    uint8_t srcGm[param.m * param.n * dataBytes];
    uint8_t dstGm[param.m * param.n * dataBytes];
    uint8_t scaleGm[param.m * dataBytes];
    uint8_t offsetGm[param.m * dataBytes];
    param.cal_func(srcGm, dstGm, scaleGm, offsetGm, param.m, param.n, param.calCount, param.gs);
}
