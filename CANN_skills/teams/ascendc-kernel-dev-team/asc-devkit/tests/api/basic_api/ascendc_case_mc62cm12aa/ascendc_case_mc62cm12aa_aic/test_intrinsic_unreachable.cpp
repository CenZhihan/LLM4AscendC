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
#include "mockcpp/mockcpp.hpp"
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "test_utils.h"

using namespace std;
using namespace AscendC;

template <typename T>
class KernelUnreachable {
public:
    __aicore__ inline KernelUnreachable()
    {}

    __aicore__ inline void RunCopyGm2CBufMultiNd2Nz(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint64_t loop1SrcStride = 32;
        uint8_t l2CacheCtl = 0;
        uint16_t nValue = 32;
        uint32_t dValue = 32;
        uint64_t loop4SrcStride = 0;
        bool smallc0En = false;
        copy_gm_to_cbuf_multi_nd2nz((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, loop1SrcStride, l2CacheCtl,
            nValue,  dValue, loop4SrcStride, smallc0En);
    }

    __aicore__ inline void RunCopyGm2UbufAlignV2(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint32_t burst_num = 1;
        uint32_t burst_len = 8;
        uint8_t left_padding_count = 0;
        uint8_t right_padding_count = 0;
        bool constant_padding_ctl = false;
        uint8_t l2_cache_ctl = 0;
        uint64_t burst_src_stride = 8;
        uint32_t burst_dst_stride = 8;
        copy_gm_to_ubuf_align_v2((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, burst_num,
            burst_len, left_padding_count, right_padding_count, constant_padding_ctl,
            l2_cache_ctl, burst_src_stride, burst_dst_stride);

    }

    __aicore__ inline void RunCopyGm2CbufAlignV2(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint32_t burst_num = 1;
        uint32_t burst_len = 8;
        uint8_t left_padding_count = 0;
        uint8_t right_padding_count = 0;
        bool constant_padding_ctl = false;
        uint8_t l2_cache_ctl = 0;
        uint64_t burst_src_stride = 8;
        uint32_t burst_dst_stride = 8;
        copy_gm_to_cbuf_align_v2((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, burst_num,
            burst_len, left_padding_count, right_padding_count, constant_padding_ctl,
            l2_cache_ctl, burst_src_stride, burst_dst_stride);
    }

    __aicore__ inline void RunImg2ColV2Cbuf2Cb(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint16_t kExtension = 1;
        uint16_t mExtension = 1;
        uint16_t kStartPt = 0;
        uint16_t mStartPt = 0;
        uint8_t strideW = 1;
        uint8_t strideH = 1;
        uint8_t Wk = 1;
        uint8_t Hk = 1;
        uint8_t dilationFilterW = 1;
        uint8_t dilationFilterH = 1;
        bool filterSizeW = false;
        bool filterSizeH = false;
        bool transpose = false;
        bool fMatrixCtrl = false;
        uint16_t sizeChannel = 1;
        // AscendC::ModelFactory::Instance().SetSprFMatrix(1000000); // 这里设置后会影响Load3DV2Pv的校验
        // img2colv2_cbuf_to_cb((__gm__ T *)dstGm, (__gm__ T *)srcGm, kExtension, mExtension, kStartPt, mStartPt,
        //     strideW, strideH, Wk, Hk, dilationFilterW, dilationFilterH, filterSizeW, filterSizeH, transpose,
        //     fMatrixCtrl, sizeChannel);
    }

    __aicore__ inline void Run(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        RunCopyGm2CBufMultiNd2Nz(dstGm, srcGm);
        RunCopyGm2UbufAlignV2(dstGm, srcGm);
        RunCopyGm2CbufAlignV2(dstGm, srcGm);
        RunImg2ColV2Cbuf2Cb(dstGm, srcGm);
    }
};


class TEST_INTRINSIC_UNREACHABLE : public testing::Test {
protected:
    void SetUp()
    {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown()
    {
        g_coreType = AscendC::MIX_TYPE;
    }
};

#define REGIST_INTRINSIC_UNREACHABLE(type)                                                    \
    TEST_F(TEST_INTRINSIC_UNREACHABLE, DATACOPY_NZ2NZ_Case_Bias_##type##_##high##_##width)      \
    {                                                                                           \
        KernelUnreachable<type> op;                                                             \
        uint8_t srcGm[100] = {0};                                                               \
        uint8_t dstGm[100] = {0};                                                               \
        op.Run(dstGm, srcGm);                                                                  \
    }


REGIST_INTRINSIC_UNREACHABLE(fp8_e5m2_t);
REGIST_INTRINSIC_UNREACHABLE(fp8_e4m3fn_t);
