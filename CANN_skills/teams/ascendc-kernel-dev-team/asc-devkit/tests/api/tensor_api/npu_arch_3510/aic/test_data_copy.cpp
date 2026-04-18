/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"

enum class CubeLayout {
    RowMajor,
    NZ,
    ColumnMajor,
    ZN,
    ND
};

enum class Prefix {
    gm,
    cbuf,
    biasbuf,
    fbuf,
};


class TEST_TENSOR_API_DATACOPY : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

// ==================== Constants ====================
constexpr int TEST_FRACTAL_FIXED = 16;
constexpr int TEST_C0_SIZE = 32;
constexpr int TEST_L12BT_UNIT = TEST_C0_SIZE;          // 64
constexpr int TEST_C2PIPE2GM_UNIT = TEST_C0_SIZE * 2;      // 128

constexpr int TestCeilDivision(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// ==================== Mocker Method ====================
// ND2ND: CopyGmToCbufNDBase
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_gm_to_cbuf_nd2nd_stub(__cbuf__ void* dst, __gm__ void* src, uint8_t sid, \
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, pad_t padValue) {
    EXPECT_EQ(sid, 0);
    EXPECT_EQ(blockCount, DST_SIZE1);
    EXPECT_EQ(blockLen, DST_SIZE2 * sizeof(DTYPE));
    EXPECT_EQ(srcStride, SRC_SIZE2 * sizeof(DTYPE));
    EXPECT_EQ(dstStride, DST_SIZE2 * sizeof(DTYPE));
}

// NZ2NZ: CopyGmToCbufNZBase
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_gm_to_cbuf_nz2nz_stub(__cbuf__ void* dst, __gm__ void* src, uint8_t sid, \
    uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, pad_t padValue) {
    EXPECT_EQ(sid, 0);
    EXPECT_EQ(blockCount, DST_SIZE2 * sizeof(DTYPE) / TEST_C0_SIZE);
    EXPECT_EQ(blockLen, DST_SIZE1 * TEST_C0_SIZE);
    EXPECT_EQ(srcStride, SRC_SIZE1 * TEST_C0_SIZE);
    EXPECT_EQ(dstStride, DST_SIZE1 * TEST_C0_SIZE);
}

// ND2NZ: CopyGmToCbufMultiND2NZBase
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_gm_to_cbuf_multi_nd2nz_stub(__cbuf__ DTYPE* dst, __gm__ DTYPE* src, uint8_t sid,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride) {
    EXPECT_EQ(sid, 0);
    EXPECT_EQ(ndNum, 1);
    EXPECT_EQ(srcNdMatrixStride, 0);
    EXPECT_EQ(dstNzMatrixStride, 0);
    EXPECT_EQ(nValue, DST_SIZE1);
    EXPECT_EQ(dValue, DST_SIZE2);
    EXPECT_EQ(srcDValue, SRC_SIZE2);
    EXPECT_EQ(dstNzC0Stride, DST_SIZE1);
    EXPECT_EQ(dstNzNStride, 1);
}

// DN2ZN: CopyGmToCbufMultiDN2ZNBase
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_gm_to_cbuf_multi_dn2zn_stub(__cbuf__ DTYPE* dst, __gm__ DTYPE* src, uint8_t sid,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride) {
    EXPECT_EQ(sid, 0);
    EXPECT_EQ(ndNum, 1);
    EXPECT_EQ(srcNdMatrixStride, 0);
    EXPECT_EQ(dstNzMatrixStride, 0);
    EXPECT_EQ(nValue, DST_SIZE2);
    EXPECT_EQ(dValue, DST_SIZE1);
    EXPECT_EQ(srcDValue, SRC_SIZE1);
    EXPECT_EQ(dstNzC0Stride, DST_SIZE2);
    EXPECT_EQ(dstNzNStride, 1);
}

// L1 -> BIAS: CopyCbufToBT3501
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_cbuf_to_bt_stub(uint64_t dst, __cbuf__ DTYPE* src, uint16_t convControl, uint16_t blockCount, uint16_t blockLen,
                                uint16_t srcStride, uint16_t dstStride) {
    EXPECT_EQ(convControl, 0);
    EXPECT_EQ(blockCount, DST_SIZE1);
    EXPECT_EQ(blockLen, DST_SIZE2 * sizeof(DTYPE) / TEST_L12BT_UNIT);
    EXPECT_EQ(srcStride, (SRC_SIZE2 - DST_SIZE2) * sizeof(DTYPE) / TEST_C0_SIZE);
    EXPECT_EQ(dstStride, (DST_SIZE2 - DST_SIZE2) * sizeof(DTYPE) / TEST_L12BT_UNIT);
}

// L1 -> BIAS two type: CopyCbufToBT3501
template<typename SRC_DTYPE, typename DST_DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_cbuf_to_bt_two_type_stub(uint64_t dst, __cbuf__ SRC_DTYPE* src, uint16_t convControl, uint16_t blockCount, uint16_t blockLen,
                                uint16_t srcStride, uint16_t dstStride) {
    if constexpr (std::is_same_v<SRC_DTYPE, half>) {
        EXPECT_EQ(convControl, 1);
    } else {
        EXPECT_EQ(convControl, 0);
    }
    EXPECT_EQ(blockCount, DST_SIZE1);
    EXPECT_EQ(blockLen, DST_SIZE2 * sizeof(DST_DTYPE) / TEST_L12BT_UNIT);
    EXPECT_EQ(srcStride, (SRC_SIZE2 - DST_SIZE2) * sizeof(SRC_DTYPE) / TEST_C0_SIZE);
    EXPECT_EQ(dstStride, (DST_SIZE2 - DST_SIZE2) * sizeof(DST_DTYPE) / TEST_L12BT_UNIT);
}


// L1 -> FIXBUF: CopyCbufToFB3501
template<typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, int DST_SIZE2>
__aicore__ inline void copy_cbuf_to_fbuf_stub(__fbuf__ void* dst, __cbuf__ void* src, uint16_t blockCount, uint16_t blockLen,
                                uint16_t srcStride, uint16_t dstStride) {
    EXPECT_EQ(blockCount, DST_SIZE1);
    EXPECT_EQ(blockLen, TestCeilDivision(DST_SIZE2 * sizeof(DTYPE), TEST_C2PIPE2GM_UNIT));
    EXPECT_EQ(srcStride, TestCeilDivision(SRC_SIZE2 * sizeof(DTYPE), TEST_C0_SIZE));
    EXPECT_EQ(dstStride, TestCeilDivision(DST_SIZE2 * sizeof(DTYPE), TEST_C2PIPE2GM_UNIT));
}

template<typename T, typename DTYPE, int SIZE1, int SIZE2, AscendC::Hardware LOCATION, CubeLayout LAYOUT>
__aicore__ inline void check_tensor_success(const T& tensor) {
    using namespace AscendC::Te;
    AscendC::Hardware pos = GetHardPos<T>();
    EXPECT_EQ(pos, LOCATION);
}

template<typename Coord>
__aicore__ inline void check_coord_success (const Coord& coord) {

}

// tensor2tensor run method
template<typename T, typename U, typename Coord, typename DTYPE, int SRC_SIZE1, int SRC_SIZE2, int DST_SIZE1, \
                    int DST_SIZE2, AscendC::Hardware SRC_LOCATION, CubeLayout SRC_LAYOUT, AscendC::Hardware DST_LOCATION, CubeLayout DST_LAYOUT>
__aicore__ inline void tensor2tensor_run_stub(const T& dst, const U& src, const Coord& coord) {
    check_tensor_success<U, DTYPE, SRC_SIZE1, SRC_SIZE2, SRC_LOCATION, SRC_LAYOUT>(src);
    check_tensor_success<T, DTYPE, DST_SIZE1, DST_SIZE2, DST_LOCATION, DST_LAYOUT>(dst);
    check_coord_success<Coord>(coord);
}

// create tensor
#define CREATE_TENSOR(DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2, SRC_PREFIX, SRC_LOCATION, SRC_LAYOUT, DST_PREFIX, DST_LOCATION, DST_LAYOUT) \
    using namespace AscendC::Te; \
    __##SRC_PREFIX##__ DTYPE srcData[SRC_SIZE1 * SRC_SIZE2 * sizeof(DTYPE)]; \
    __##DST_PREFIX##__ DTYPE dstData[DST_SIZE1 * DST_SIZE2 * sizeof(DTYPE)]; \
    \
    auto srcIterator = Make##SRC_LOCATION##memPtr(srcData); \
    auto srcLayout = Make##SRC_LAYOUT##Layout<DTYPE>(SRC_SIZE1, SRC_SIZE2); \
    auto srcTensor = MakeTensor(srcIterator, srcLayout); \
    \
    auto dstIterator = Make##DST_LOCATION##memPtr(dstData); \
    auto dstLayout = Make##DST_LAYOUT##Layout<DTYPE>(DST_SIZE1, DST_SIZE2); \
    auto dstTensor = MakeTensor(dstIterator, dstLayout);


// create tensor
#define CREATE_TENSOR_TWO_TYPE(SRC_DTYPE, SRC_SIZE1, SRC_SIZE2, DST_DTYPE, DST_SIZE1, DST_SIZE2, SRC_PREFIX, SRC_LOCATION, SRC_LAYOUT, DST_PREFIX, DST_LOCATION, DST_LAYOUT) \
    using namespace AscendC::Te; \
    __##SRC_PREFIX##__ SRC_DTYPE srcData[SRC_SIZE1 * SRC_SIZE2 * sizeof(SRC_DTYPE)]; \
    __##DST_PREFIX##__ DST_DTYPE dstData[DST_SIZE1 * DST_SIZE2 * sizeof(DST_DTYPE)]; \
    \
    auto srcIterator = Make##SRC_LOCATION##memPtr(srcData); \
    auto srcLayout = Make##SRC_LAYOUT##Layout<SRC_DTYPE>(SRC_SIZE1, SRC_SIZE2); \
    auto srcTensor = MakeTensor(srcIterator, srcLayout); \
    \
    auto dstIterator = Make##DST_LOCATION##memPtr(dstData); \
    auto dstLayout = Make##DST_LAYOUT##Layout<DST_DTYPE>(DST_SIZE1, DST_SIZE2); \
    auto dstTensor = MakeTensor(dstIterator, dstLayout);

// ==================== Test case ====================

// L1 to BIAS ND2ND test case
#define DATA_COPY_TEST_L12BIAS_ND2ND(DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2) \
    TEST_F(TEST_TENSOR_API_DATACOPY, TEST_TENSOR_API_DATACOPY_L12BIAS_ND2ND_##DTYPE##_SRC_SIZE##_##SRC_SIZE1##x##SRC_SIZE2##_DST_SIZE##_##DST_SIZE1##x##DST_SIZE2) \
    { \
        using namespace AscendC::Te; \
        MOCKER_CPP(copy_cbuf_to_bt, void(uint64_t, __cbuf__ DTYPE*, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t)) \
            .times(1) \
            .will(invoke(&copy_cbuf_to_bt_stub<DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2>)); \
        CREATE_TENSOR(DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2, cbuf, L1, ND, biasbuf, Bias, ND) \
        DataCopy<DEFAULT_DATA_COPY_TRAIT>(dstTensor, srcTensor); \
        GlobalMockObject::verify(); \
    }

DATA_COPY_TEST_L12BIAS_ND2ND(float, 1, 64, 1, 64)
DATA_COPY_TEST_L12BIAS_ND2ND(int32_t, 1, 64, 1, 64)

// L1 to BIAS two data type  test case
#define DATA_COPY_TEST_L12BIAS_TWO_TYPE_ND2ND(SRC_DTYPE, SRC_SIZE1, SRC_SIZE2, DST_DTYPE, DST_SIZE1, DST_SIZE2) \
    TEST_F(TEST_TENSOR_API_DATACOPY, TEST_TENSOR_API_DATACOPY_L12BIAS_TWO_TYPE_ND2ND_##SRC_DTYPE##_SRC_SIZE##_##SRC_SIZE1##x##SRC_SIZE2##DST_DTYPE##_DST_SIZE##_##DST_SIZE1##x##DST_SIZE2) \
    { \
        using namespace AscendC::Te; \
        MOCKER_CPP(copy_cbuf_to_bt, void(uint64_t, __cbuf__ SRC_DTYPE*, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t)) \
            .times(1) \
            .will(invoke(&copy_cbuf_to_bt_two_type_stub<SRC_DTYPE, DST_DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2>)); \
        CREATE_TENSOR_TWO_TYPE(SRC_DTYPE, SRC_SIZE1, SRC_SIZE2, DST_DTYPE, DST_SIZE1, DST_SIZE2, cbuf, L1, ND, biasbuf, Bias, ND) \
        DataCopy<DEFAULT_DATA_COPY_TRAIT>(dstTensor, srcTensor); \
        GlobalMockObject::verify(); \
    }

DATA_COPY_TEST_L12BIAS_TWO_TYPE_ND2ND(bfloat16_t, 1, 64, float, 1, 64)
DATA_COPY_TEST_L12BIAS_TWO_TYPE_ND2ND(half, 1, 64, float, 1, 64)


// L1 to FP ND2ND test case
#define DATA_COPY_TEST_L12FB_ND2ND(DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2) \
    TEST_F(TEST_TENSOR_API_DATACOPY, TEST_TENSOR_API_DATACOPY_L12FB_ND2ND_##DTYPE##_SRC_SIZE##_##SRC_SIZE1##x##SRC_SIZE2##_DST_SIZE##_##DST_SIZE1##x##DST_SIZE2) \
    { \
        using namespace AscendC::Te; \
        MOCKER_CPP(copy_cbuf_to_fbuf, void(__fbuf__ void*, __cbuf__ void*, uint16_t, uint16_t, uint16_t, uint16_t)) \
            .times(1) \
            .will(invoke(&copy_cbuf_to_fbuf_stub<DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2>)); \
        CREATE_TENSOR(DTYPE, SRC_SIZE1, SRC_SIZE2, DST_SIZE1, DST_SIZE2, cbuf, L1, ND, fbuf, Fixbuf, ND) \
        DataCopy<DEFAULT_DATA_COPY_TRAIT>(dstTensor, srcTensor); \
        GlobalMockObject::verify(); \
    }

DATA_COPY_TEST_L12FB_ND2ND(uint64_t, 1, 64, 1, 64)
