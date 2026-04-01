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
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

constexpr int THREAD_DIM = 128;

template <typename T>
class KernelWarpLevelCompute {
    public:
        __aicore__ KernelWarpLevelCompute() {}
        // int32_t
        __aicore__ inline void Process(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
        // uint32_t
        __aicore__ inline void Process2(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
        // float
        __aicore__ inline void Process3(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
        // int64_t
        __aicore__ inline void Process4(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
        // uint64_t
        __aicore__ inline void Process5(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width);
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < THREAD_DIM; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        int32_t value = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = asc_all(pred);
                break;
            case 1:
                dst[idx] = asc_any(pred);
                break;
            case 2:
                dst[idx] = asc_ballot(pred);
                break;
            case 3:
                dst[idx] = asc_activemask();
                break;
            case 4:
                dst[idx] = asc_shfl(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = asc_shfl_up(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = asc_shfl_down(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = asc_shfl_xor(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = asc_reduce_add(value);
                break;
            case 9:
                dst[idx] = asc_reduce_max(value);
                break;
            case 10:
                dst[idx] = asc_reduce_min(value);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    asc_vf_call<KernelWarpLevelComputeCompute<T>>(dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute2(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < THREAD_DIM; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        uint32_t value = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = asc_all(pred);
                break;
            case 1:
                dst[idx] = asc_any(pred);
                break;
            case 2:
                dst[idx] = asc_ballot(pred);
                break;
            case 3:
                dst[idx] = asc_activemask();
                break;
            case 4:
                dst[idx] = asc_shfl(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = asc_shfl_up(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = asc_shfl_down(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = asc_shfl_xor(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = asc_reduce_add(value);
                break;
            case 9:
                dst[idx] = asc_reduce_max(value);
                break;
            case 10:
                dst[idx] = asc_reduce_min(value);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process2(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    asc_vf_call<KernelWarpLevelComputeCompute2<T>>(dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute3(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < THREAD_DIM; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        float value = laneId;
        int32_t value2 = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = asc_all(pred);
                break;
            case 1:
                dst[idx] = asc_any(pred);
                break;
            case 2:
                dst[idx] = asc_ballot(pred);
                break;
            case 3:
                dst[idx] = asc_activemask();
                break;
            case 4:
                dst[idx] = asc_shfl(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = asc_shfl_up(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = asc_shfl_down(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = asc_shfl_xor(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = asc_reduce_add(value2);
                break;
            case 9:
                dst[idx] = asc_reduce_max(value2);
                break;
            case 10:
                dst[idx] = asc_reduce_min(value2);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process3(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    asc_vf_call<KernelWarpLevelComputeCompute3<T>>(dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute4(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < THREAD_DIM; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        int64_t value = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = asc_all(pred);
                break;
            case 1:
                dst[idx] = asc_any(pred);
                break;
            case 2:
                dst[idx] = asc_ballot(pred);
                break;
            case 3:
                dst[idx] = asc_activemask();
                break;
            case 4:
                dst[idx] = asc_shfl(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = asc_shfl_up(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = asc_shfl_down(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = asc_shfl_xor(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = asc_reduce_add((int32_t)value);
                break;
            case 9:
                dst[idx] = asc_reduce_max((int32_t)value);
                break;
            case 10:
                dst[idx] = asc_reduce_min((int32_t)value);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process4(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    asc_vf_call<KernelWarpLevelComputeCompute4<T>>(dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(1024) inline __aicore__ void KernelWarpLevelComputeCompute5(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    for (int idx = AscendC::Simt::GetThreadIdx<0>() + block_idx * AscendC::Simt::GetThreadNum<0>(); idx < THREAD_DIM; idx += block_num * AscendC::Simt::GetThreadNum<0>()) {
        int32_t pred = predicate != -1 ? predicate : idx%32;
        int32_t laneId = idx % 32;
        uint64_t value = laneId;
        int32_t widthc = width != 0 ? width : 32;

        switch (mode) {
            case 0:
                dst[idx] = asc_all(pred);
                break;
            case 1:
                dst[idx] = asc_any(pred);
                break;
            case 2:
                dst[idx] = asc_ballot(pred);
                break;
            case 3:
                dst[idx] = asc_activemask();
                break;
            case 4:
                dst[idx] = asc_shfl(value, laneOp, widthc);
                break;
            case 5:
                dst[idx] = asc_shfl_up(value, laneOp, widthc);
                break;
            case 6:
                dst[idx] = asc_shfl_down(value, laneOp, widthc);
                break;
            case 7:
                dst[idx] = asc_shfl_xor(value, laneOp, widthc);
                break;
            case 8:
                dst[idx] = asc_reduce_add((int32_t)value);
                break;
            case 9:
                dst[idx] = asc_reduce_max((int32_t)value);
                break;
            case 10:
                dst[idx] = asc_reduce_min((int32_t)value);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelWarpLevelCompute<T>::Process5(const int mode, __gm__ T* dst, const int predicate, const int laneOp, const int width)
{
    asc_vf_call<KernelWarpLevelComputeCompute5<T>>(dim3(THREAD_DIM, 1, 1), mode, dst, predicate, laneOp, width);
}

// ================================ Test int32_t start ================================
struct WarpLevelcComputeParams_int32 {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelcComputeTestsuite_int32 : public testing::Test, public testing::WithParamInterface<WarpLevelcComputeParams_int32> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelcComputeTestCase_int32, WarpLevelcComputeTestsuite_int32,
    ::testing::Values(WarpLevelcComputeParams_int32 {.mode = 0, .predicate = -1},
        WarpLevelcComputeParams_int32 {.mode = 0, .predicate = 0},
        WarpLevelcComputeParams_int32 {.mode = 0, .predicate = 1},

        WarpLevelcComputeParams_int32 {.mode = 1, .predicate = -1},
        WarpLevelcComputeParams_int32 {.mode = 1, .predicate = 0},
        WarpLevelcComputeParams_int32 {.mode = 1, .predicate = 1},

        WarpLevelcComputeParams_int32 {.mode = 2, .predicate = -1},
        WarpLevelcComputeParams_int32 {.mode = 2, .predicate = 0},
        WarpLevelcComputeParams_int32 {.mode = 2, .predicate = 1},

        WarpLevelcComputeParams_int32 {.mode = 3},

        WarpLevelcComputeParams_int32 {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int32 {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int32 {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int32 {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int32 {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int32 {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int32 {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int32 {.mode = 7, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int32 {.mode = 8},

        WarpLevelcComputeParams_int32 {.mode = 9},

        WarpLevelcComputeParams_int32 {.mode = 10}
    ));

TEST_P(WarpLevelcComputeTestsuite_int32, WarpLevelcComputeTestCase_int32)
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

    int32_t expectValues[shapeSize] = {0};
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
// ================================ Test int32_t end ================================

// ================================ Test uint32_t start ================================
struct WarpLevelcComputeParams_uint32 {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelcComputeTestsuite_uint32 : public testing::Test, public testing::WithParamInterface<WarpLevelcComputeParams_uint32> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelcComputeTestCase_uint32, WarpLevelcComputeTestsuite_uint32,
    ::testing::Values(WarpLevelcComputeParams_uint32 {.mode = 0, .predicate = -1},
        WarpLevelcComputeParams_uint32 {.mode = 0, .predicate = 0},
        WarpLevelcComputeParams_uint32 {.mode = 0, .predicate = 1},

        WarpLevelcComputeParams_uint32 {.mode = 1, .predicate = -1},
        WarpLevelcComputeParams_uint32 {.mode = 1, .predicate = 0},
        WarpLevelcComputeParams_uint32 {.mode = 1, .predicate = 1},

        WarpLevelcComputeParams_uint32 {.mode = 2, .predicate = -1},
        WarpLevelcComputeParams_uint32 {.mode = 2, .predicate = 0},
        WarpLevelcComputeParams_uint32 {.mode = 2, .predicate = 1},

        WarpLevelcComputeParams_uint32 {.mode = 3},

        WarpLevelcComputeParams_uint32 {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint32 {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint32 {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint32 {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint32 {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint32 {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint32 {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint32 {.mode = 7, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint32 {.mode = 8},

        WarpLevelcComputeParams_uint32 {.mode = 9},

        WarpLevelcComputeParams_uint32 {.mode = 10}
    ));

TEST_P(WarpLevelcComputeTestsuite_uint32, WarpLevelcComputeTestCase_uint32)
{
    auto param = GetParam();

    int typeSize = 4;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelWarpLevelCompute<uint32_t> op;
    op.Process2(param.mode, (__gm__ uint32_t*)dstGM, param.predicate, param.laneOp, param.width);

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
// ================================ Test uint32_t end ================================

// ================================ Test float start ================================
struct WarpLevelcComputeParams_float {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelcComputeTestsuite_float : public testing::Test, public testing::WithParamInterface<WarpLevelcComputeParams_float> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelcComputeTestCase_float, WarpLevelcComputeTestsuite_float,
    ::testing::Values(WarpLevelcComputeParams_float {.mode = 0, .predicate = -1},
        WarpLevelcComputeParams_float {.mode = 0, .predicate = 0},
        WarpLevelcComputeParams_float {.mode = 0, .predicate = 1},

        WarpLevelcComputeParams_float {.mode = 1, .predicate = -1},
        WarpLevelcComputeParams_float {.mode = 1, .predicate = 0},
        WarpLevelcComputeParams_float {.mode = 1, .predicate = 1},

        WarpLevelcComputeParams_float {.mode = 2, .predicate = -1},
        WarpLevelcComputeParams_float {.mode = 2, .predicate = 0},
        WarpLevelcComputeParams_float {.mode = 2, .predicate = 1},

        WarpLevelcComputeParams_float {.mode = 3},

        WarpLevelcComputeParams_float {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_float {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_float {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_float {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_float {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_float {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_float {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_float {.mode = 7, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_float {.mode = 8},

        WarpLevelcComputeParams_float {.mode = 9},

        WarpLevelcComputeParams_float {.mode = 10}
    ));

TEST_P(WarpLevelcComputeTestsuite_float, WarpLevelcComputeTestCase_float)
{
    auto param = GetParam();

    int typeSize = 4;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelWarpLevelCompute<float> op;
    op.Process3(param.mode, (__gm__ float*)dstGM, param.predicate, param.laneOp, param.width);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    float expectValues[shapeSize] = {0};
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

    for (int i = 0; i < shapeSize; i++) {
        uint32_t expectBits;
        memcpy(&expectBits, &expectValues[i], sizeof(float));

        ASSERT_EQ(static_cast<uint8_t>(expectBits & 0xFF), dstGM[i * typeSize]);
        ASSERT_EQ(static_cast<uint8_t>((expectBits >> 8) & 0xFF), dstGM[i * typeSize + 1]);
        ASSERT_EQ(static_cast<uint8_t>((expectBits >> 16) & 0xFF), dstGM[i * typeSize + 2]);
        ASSERT_EQ(static_cast<uint8_t>((expectBits >> 24) & 0xFF), dstGM[i * typeSize + 3]);
        
    }
}
// ================================ Test float end ================================

// ================================ Test int64_t start ================================
struct WarpLevelcComputeParams_int64t {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelcComputeTestsuite_int64t : public testing::Test, public testing::WithParamInterface<WarpLevelcComputeParams_int64t> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelcComputeTestCase_int64t, WarpLevelcComputeTestsuite_int64t,
    ::testing::Values(WarpLevelcComputeParams_int64t {.mode = 0, .predicate = -1},
        WarpLevelcComputeParams_int64t {.mode = 0, .predicate = 0},
        WarpLevelcComputeParams_int64t {.mode = 0, .predicate = 1},

        WarpLevelcComputeParams_int64t {.mode = 1, .predicate = -1},
        WarpLevelcComputeParams_int64t {.mode = 1, .predicate = 0},
        WarpLevelcComputeParams_int64t {.mode = 1, .predicate = 1},

        WarpLevelcComputeParams_int64t {.mode = 2, .predicate = -1},
        WarpLevelcComputeParams_int64t {.mode = 2, .predicate = 0},
        WarpLevelcComputeParams_int64t {.mode = 2, .predicate = 1},

        WarpLevelcComputeParams_int64t {.mode = 3},

        WarpLevelcComputeParams_int64t {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int64t {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int64t {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int64t {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int64t {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int64t {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int64t {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_int64t {.mode = 7, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_int64t {.mode = 8},

        WarpLevelcComputeParams_int64t {.mode = 9},

        WarpLevelcComputeParams_int64t {.mode = 10}
    ));

TEST_P(WarpLevelcComputeTestsuite_int64t, WarpLevelcComputeTestCase_int64t)
{
    auto param = GetParam();

    int typeSize = 8;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelWarpLevelCompute<int64_t> op;
    op.Process4(param.mode, (__gm__ int64_t*)dstGM, param.predicate, param.laneOp, param.width);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    int64_t expectValues[shapeSize] = {0};
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
// ================================ Test int64_t end ================================

// ================================ Test uint64_t start ================================
struct WarpLevelcComputeParams_uint64t {
    int32_t mode = 0;

    // for Warp vote function
    int32_t predicate = 0;

    // for Warp reduce and shuffle function
    int32_t laneOp = 0;
    int32_t width = 0;
};

class WarpLevelcComputeTestsuite_uint64t : public testing::Test, public testing::WithParamInterface<WarpLevelcComputeParams_uint64t> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(WarpLevelcComputeTestCase_uint64t, WarpLevelcComputeTestsuite_uint64t,
    ::testing::Values(WarpLevelcComputeParams_uint64t {.mode = 0, .predicate = -1},
        WarpLevelcComputeParams_uint64t {.mode = 0, .predicate = 0},
        WarpLevelcComputeParams_uint64t {.mode = 0, .predicate = 1},

        WarpLevelcComputeParams_uint64t {.mode = 1, .predicate = -1},
        WarpLevelcComputeParams_uint64t {.mode = 1, .predicate = 0},
        WarpLevelcComputeParams_uint64t {.mode = 1, .predicate = 1},

        WarpLevelcComputeParams_uint64t {.mode = 2, .predicate = -1},
        WarpLevelcComputeParams_uint64t {.mode = 2, .predicate = 0},
        WarpLevelcComputeParams_uint64t {.mode = 2, .predicate = 1},

        WarpLevelcComputeParams_uint64t {.mode = 3},

        WarpLevelcComputeParams_uint64t {.mode = 4, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint64t {.mode = 4, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint64t {.mode = 5, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint64t {.mode = 5, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint64t {.mode = 6, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint64t {.mode = 6, .laneOp = 1, .width = 32},

        WarpLevelcComputeParams_uint64t {.mode = 7, .laneOp = 1, .width = 16},
        WarpLevelcComputeParams_uint64t {.mode = 7, .laneOp = 1, .width = 32}
    ));

TEST_P(WarpLevelcComputeTestsuite_uint64t, WarpLevelcComputeTestCase_uint64t)
{
    auto param = GetParam();

    int typeSize = 8;
    int shapeSize = THREAD_DIM;
    uint8_t dstGM[shapeSize * typeSize] = {0};

    KernelWarpLevelCompute<uint64_t> op;
    op.Process5(param.mode, (__gm__ uint64_t*)dstGM, param.predicate, param.laneOp, param.width);

    for (int i = 0; i < 256; i += 2) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("%02x%02x ", dstGM[i], dstGM[i+1]);
    }

    uint64_t expectValues[shapeSize] = {0};
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
// ================================ Test uint64_t end ================================