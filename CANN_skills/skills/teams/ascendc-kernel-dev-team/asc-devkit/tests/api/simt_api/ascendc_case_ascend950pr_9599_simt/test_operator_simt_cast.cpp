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
#include <type_traits>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

#define THREAD_DIM 128

template <typename DstType, typename SrcType>
class KernelCast {
public:
    __aicore__ KernelCast() {}

public:
    __aicore__ inline void Process(__gm__ DstType* dst, const int mode);
};

template <typename DstType, typename SrcType>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelCastCompute(__gm__ DstType* dst, const int mode)
{
    for (int idx = GetThreadIdx<0>() + block_idx * GetThreadNum<0>(); idx < 128;
         idx += block_num * GetThreadNum<0>()) {

        if (mode == 0) {
            half srcNum = static_cast<half>(idx);
            if (idx == 1022) {
                srcNum = static_cast<half>(-30144);
            }
            if (idx == 1023) {
                srcNum = static_cast<half>(-87825);
            }
            dst[idx] = Cast<DstType, half, RoundMode::CAST_NONE>(srcNum);
        } else if (mode == 1) {
            DstType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Rint(srcNum);
        } else if (mode == 2) {
            DstType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Floor(srcNum);
        } else if (mode == 3) {
            DstType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Ceil(srcNum);
        } else if (mode == 4) {
            SrcType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Cast<DstType, SrcType, RoundMode::CAST_FLOOR>(srcNum);
        } else if (mode == 5) {
            SrcType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Cast<DstType, SrcType, RoundMode::CAST_CEIL>(srcNum);
        } else if (mode == 6) {
            DstType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Round(srcNum);
        } else if (mode == 7) {
            DstType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Trunc(srcNum);
        } else if (mode == 8) {
            SrcType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Cast<DstType, SrcType, RoundMode::CAST_EVEN>(srcNum);
        } else if (mode == 9) {
            SrcType srcNum = idx;
            if (idx == 1022) {
                srcNum = -30144;
            }
            if (idx == 1023) {
                srcNum = -87825;
            }
            dst[idx] = Cast<DstType, SrcType, RoundMode::CAST_ZERO>(srcNum);
        }
    }
}

template <typename DstType, typename SrcType>
__aicore__ inline void KernelCast<DstType, SrcType>::Process(__gm__ DstType* dst, const int mode)
{
    AscendC::Simt::VF_CALL<KernelCastCompute<DstType, SrcType>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), dst, mode);
}

struct CastParams {
    int32_t mode;
};

class CastTestsuite : public testing::Test, public testing::WithParamInterface<CastParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(CastTestCase, CastTestsuite,
                        ::testing::Values(CastParams{0}, CastParams{1}, CastParams{2}, CastParams{3}, CastParams{4},
                                          CastParams{5}, CastParams{6}, CastParams{7}, CastParams{8}, CastParams{9}));

TEST_P(CastTestsuite, CastTestCase)
{
    auto param = GetParam();
    int32_t mode = mode = param.mode;
    int fpByteSize = 4;
    int shapeSize = 128;

    uint8_t dstGm[shapeSize * fpByteSize] = {0};
    KernelCast<float, int> op;
    op.Process((__gm__ float*)dstGm, mode);

    float expectValues[shapeSize] = {0};
    for (int i = 0; i < shapeSize; i += 1) {
        expectValues[i] = i;
        if (i == 1022) {
            expectValues[i] = -30144;
        }
        if (i == 1023) {
            expectValues[i] = -87825;
        }
        if (mode == 0) {
            if (i == 1023) {
                expectValues[i] =  half(-87825);
            }
            continue;
        } else if (mode == 1) {
            expectValues[i] = rint(expectValues[i]);
        } else if (mode == 2) {
            expectValues[i] = floor(expectValues[i]);
        } else if (mode == 3) {
            expectValues[i] = ceil(expectValues[i]);
        } else if (mode == 4) {
            expectValues[i] = floor(expectValues[i]);
        } else if (mode == 5) {
            expectValues[i] = ceil(expectValues[i]);
        } else if (mode == 6) {
            expectValues[i] = rint(expectValues[i]);
        } else if (mode == 7) {
            expectValues[i] = round(expectValues[i]);
        } else if (mode == 8) {
            expectValues[i] = rint(expectValues[i]);
        } else if (mode == 9) {
            expectValues[i] = trunc(expectValues[i]);
        }
    }

    float *actualValue = reinterpret_cast<float *>(dstGm);
    for (int i = 0; i < shapeSize; i += 1) {
        ASSERT_EQ(expectValues[i], actualValue[i]);
        ASSERT_EQ(expectValues[i], actualValue[i]);
        ASSERT_EQ(expectValues[i], actualValue[i]);
        ASSERT_EQ(expectValues[i], actualValue[i]);
    }
}
