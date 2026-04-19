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
// #include "api_check/kernel_cpu_check.h"

using namespace std;
using namespace AscendC;

struct TestDataCopySliceParams {
    uint32_t srcdataSize;
    uint32_t dstdataSize;
    uint32_t typeSize;
    SliceInfo srcSliceInfoList[2];
    SliceInfo dstSliceInfoList[2];
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, SliceInfo[], SliceInfo[]);
};

class TestDataCopySliceSuite : public testing::Test, public testing::WithParamInterface<TestDataCopySliceParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T>
void main_data_copy_slice_kernel(__gm__ uint8_t *__restrict__ srcGm, __gm__ uint8_t *__restrict__ dstGm,
    __gm__ uint32_t dstdataSize, __gm__ uint32_t srcdataSize, __gm__ SliceInfo dstSliceInfo[],
    __gm__ SliceInfo srcSliceInfo[])
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;


    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcdataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstdataSize);


    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dstdataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    DataCopy(inputLocal, srcGlobal, dstSliceInfo, srcSliceInfo, 2);
    DataCopy(dstGlobal, inputLocal, dstSliceInfo, dstSliceInfo, 2);
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_SLICE, TestDataCopySliceSuite,
    ::testing::Values(
        TestDataCopySliceParams{264, 96, 4, {{16, 71, 8, 3, 88}, {0, 2, 1, 1, 3}}, {{0, 47, 0, 3, 48}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int32_t>},
        TestDataCopySliceParams{264, 96, 4, {{16, 71, 8, 3, 88}, {0, 2, 1, 1, 3}}, {{0, 47, 0, 3, 48}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<float>},
        TestDataCopySliceParams{261, 96, 4, {{16, 70, 7, 3, 87}, {0, 2, 1, 1, 3}}, {{0, 47, 0, 3, 48}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int32_t>},
        TestDataCopySliceParams{261, 96, 4, {{16, 70, 7, 3, 87}, {0, 2, 1, 1, 3}}, {{0, 47, 0, 3, 48}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<float>},
        TestDataCopySliceParams{240, 64, 2, {{16, 63, 16, 1, 80}, {0, 2, 1, 1, 3}}, {{0, 31, 0, 1, 32}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<half>},
        TestDataCopySliceParams{240, 64, 2, {{16, 63, 16, 1, 80}, {0, 2, 1, 1, 3}}, {{0, 31, 0, 1, 32}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int16_t>},
        TestDataCopySliceParams{237, 64, 2, {{16, 62, 15, 1, 80}, {0, 2, 1, 1, 3}}, {{0, 31, 0, 1, 32}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<half>},
        TestDataCopySliceParams{237, 64, 2, {{16, 62, 15, 1, 80}, {0, 2, 1, 1, 3}}, {{0, 31, 0, 1, 32}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int16_t>},
        TestDataCopySliceParams{384, 128, 1, {{16, 111, 32, 1, 128}, {0, 2, 1, 1, 3}}, {{0, 63, 0, 1, 64}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int8_t>},
        TestDataCopySliceParams{384, 128, 1, {{16, 111, 32, 1, 128}, {0, 2, 1, 1, 3}}, {{0, 63, 0, 1, 64}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<uint8_t>},
        TestDataCopySliceParams{381, 128, 1, {{16, 110, 31, 1, 127}, {0, 2, 1, 1, 3}}, {{0, 63, 0, 1, 64}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<int8_t>},
        TestDataCopySliceParams{381, 128, 1, {{16, 110, 31, 1, 127}, {0, 2, 1, 1, 3}}, {{0, 63, 0, 1, 64}, {0, 1, 0, 1, 2}}, main_data_copy_slice_kernel<uint8_t>}
    ));

TEST_P(TestDataCopySliceSuite, TestDataCopySliceCases)
{
    int32_t gCoreTypeTemp = g_coreType;
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    uint8_t srcGm[param.srcdataSize * param.typeSize];
    uint8_t dstGm[param.dstdataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dstdataSize, param.srcdataSize, param.dstSliceInfoList, param.srcSliceInfoList);
    for (uint32_t i = 0; i < param.dstdataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    AscendC::SetGCoreType(gCoreTypeTemp);
}
