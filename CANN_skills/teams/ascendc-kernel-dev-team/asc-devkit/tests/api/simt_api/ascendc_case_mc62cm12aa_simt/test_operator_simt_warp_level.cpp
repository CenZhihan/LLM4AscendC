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
using namespace AscendC::Simt;

#define THREAD_DIM 128

template <typename T>
class KernelWarpLevelCompute {
    public:
        __aicore__ KernelWarpLevelCompute() {}
        __aicore__ inline void Process(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx=GetThreadIdx<0>()+block_idx*GetThreadNum<0>(); idx < THREAD_DIM; idx+=block_num*GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        T value = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = WarpAllSync(pred);
                break;
            case 1:
                dst[idx] = WarpAnySync(pred);
                break;
            case 2:
                dst[idx] = WarpBallotSync(pred);
                break;
            case 3:
                dst[idx] = WarpActiveMask();
                break;
            case 4:
                dst[idx] = WarpShflSync(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = WarpShflUpSync(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = WarpShflDownSync(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = WarpShflXorSync(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = WarpReduceAddSync(value);
                break;
            case 9:
                dst[idx] = WarpReduceMaxSync(value);
                break;
            case 10:
                dst[idx] = WarpReduceMinSync(value);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    AscendC::Simt::VF_CALL<KernelWarpLevelComputeCompute<T>>(AscendC::Simt::Dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

struct WarpLevelComputeParams {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelComputeTestsuite : public testing::Test, public testing::WithParamInterface<WarpLevelComputeParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelComputeTestCase, WarpLevelComputeTestsuite,
    ::testing::Values(WarpLevelComputeParams {.mode = 0, .predicate = -1},
        WarpLevelComputeParams {.mode = 0, .predicate = 0},
        WarpLevelComputeParams {.mode = 0, .predicate = 1},

        WarpLevelComputeParams {.mode = 1, .predicate = -1},
        WarpLevelComputeParams {.mode = 1, .predicate = 0},
        WarpLevelComputeParams {.mode = 1, .predicate = 1},

        WarpLevelComputeParams {.mode = 2, .predicate = -1},
        WarpLevelComputeParams {.mode = 2, .predicate = 0},
        WarpLevelComputeParams {.mode = 2, .predicate = 1},

        WarpLevelComputeParams {.mode = 3},

        WarpLevelComputeParams {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelComputeParams {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelComputeParams {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelComputeParams {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelComputeParams {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelComputeParams {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelComputeParams {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelComputeParams {.mode = 7, .laneOp = 1, .width = 32},

        WarpLevelComputeParams {.mode = 8},

        WarpLevelComputeParams {.mode = 9},

        WarpLevelComputeParams {.mode = 10}
    ));

TEST_P(WarpLevelComputeTestsuite, WarpLevelComputeTestCase)
{
    auto param = GetParam();

    int typeSize = 4;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelWarpLevelCompute<int32_t> op;
    op.Process(param.mode, (__gm__ int32_t*)dstGM, param.predicate, param.laneOp, param.width);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    uint32_t expectValues[shapeSize] = {0};
    switch (param.mode) {
        case 0:
            for (int i = 0; i < shapeSize; i += 1) {
                if (param.predicate == -1 || param.predicate == 0) {
                    expectValues[i] = 0;
                }
                if (param.predicate > 0) {
                    expectValues[i] = 1;
                }
            }
            break;
        case 1:
            for (int i = 0; i < shapeSize; i += 1) {
                if (param.predicate == 0) {
                    expectValues[i] = 0;
                } else {
                    expectValues[i] = 1;
                }
            }
            break;
        case 2:
            for (int i = 0; i < shapeSize; i += 1) {
                if (param.predicate == -1) {
                    expectValues[i] = 0xfffffffe;
                }
                if (param.predicate == 0) {
                    expectValues[i] = 0;
                }
                if (param.predicate > 0) {
                    expectValues[i] = 0xffffffff;
                }
            }
            break;
        case 3:
            for (int i = 0; i < shapeSize; i += 1) {
                expectValues[i] = 0xffffffff;
            }
            break;
        case 4:
            for (int i = 0; i < shapeSize; i += 1) {
                int32_t laneId = i % 32;
                expectValues[i] = laneId / param.width * param.width + param.laneOp % param.width;
            }
            break;
        case 5:
            for (int i = 0; i < shapeSize; i += 1) {
                int32_t laneId = i % 32;
                int32_t groupStart = laneId / param.width * param.width;
                if (laneId - param.laneOp >= groupStart) {
                    expectValues[i] = laneId - param.laneOp;
                } else {
                    expectValues[i] = laneId;
                }
            }
            break;
        case 6:
            for (int i = 0; i < shapeSize; i += 1) {
                int32_t laneId = i % 32;
                int32_t groupEnd = laneId / param.width * param.width + param.width;
                if (laneId + param.laneOp < groupEnd) {
                    expectValues[i] = laneId + param.laneOp;
                } else {
                    expectValues[i] = laneId;
                }
            }
            break;
        case 7:
            for (int i = 0; i < shapeSize; i += 1) {
                int32_t laneId = i % 32;
                expectValues[i] = laneId ^ param.laneOp;
            }
            break;
        case 8:
            for (int i = 0; i < shapeSize; i += 1) {
                expectValues[i] = 496;
            }
            break;
        case 9:
            for (int i = 0; i < shapeSize; i += 1) {
                expectValues[i] = 31;
            }
            break;
        case 10:
            //  expect value is 0
            break;
        default:
            break;
    }

    for (int i = 0; i < shapeSize; i += 1) {
        ASSERT_EQ(static_cast<uint8_t>(expectValues[i] & 0xFF), dstGM[i*typeSize]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 8) & 0xFF), dstGM[i*typeSize+1]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 16) & 0xFF), dstGM[i*typeSize+2]);
        ASSERT_EQ(static_cast<uint8_t>((expectValues[i] >> 24) & 0xFF), dstGM[i*typeSize+3]);
    }
}
