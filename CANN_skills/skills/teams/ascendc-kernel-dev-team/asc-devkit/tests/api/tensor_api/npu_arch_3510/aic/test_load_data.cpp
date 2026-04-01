#include <gtest/gtest.h>
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"
#include <mockcpp/mockcpp.hpp>

namespace {
    template<bool transpose>
    struct LoadDataTraitHolder {
        static constexpr AscendC::Te::LoadDataTrait trait = AscendC::Te::LoadDataTrait(transpose);
    };
}

class TEST_TENSOR_API_LOAD_DATA : public testing::Test {
protected:
    void SetUp() override {
        AscendC::SetGCoreType(1);
    }

    void TearDown() override {
        AscendC::SetGCoreType(0);
    }
};

template<bool transpose, typename T, int M_STEP, int K_STEP>
void load_cbuf_to_ca_stub(__ca__ T*dst, __cbuf__ T* src,
                          uint16_t mStartPosition, uint16_t kStartPosition,
                          uint8_t mStep, uint8_t kStep,
                          int16_t srcStride, uint16_t dstStride,
                          bool transposed) {
    EXPECT_EQ(mStep, M_STEP);
    EXPECT_EQ(kStep, K_STEP);
    EXPECT_EQ(transposed, transpose);
}

template<bool transpose, typename T, int M_STEP, int K_STEP>
void load_cbuf_to_cb_stub(__cb__ T*dst, __cbuf__ T* src,
                          uint16_t mStartPosition, uint16_t kStartPosition,
                          uint8_t mStep, uint8_t kStep,
                          int16_t srcStride, uint16_t dstStride,
                          bool transposed) {
    EXPECT_EQ(mStep, M_STEP);
    EXPECT_EQ(kStep, K_STEP);
    EXPECT_EQ(transposed, transpose);
}

#define TEST_TENSOR_API_LOAD_DATA(TYPE, M, N, SRC_FORMAT, DST_FORMAT, SRC_POS, DST_POS, SRC_TAG, DST_TAG, TRANSPOSE, COORD_I, COORD_J) \
TEST_F(TEST_TENSOR_API_LOAD_DATA, TestLoadData_##TYPE##M##N##SRC_FORMAT##DST_FORMAT##SRC_POS##DST_POS##SRC_TAG##DST_TAG##TRANSPOSE##COORD_I##COORD_J) { \
    using namespace AscendC::Te; \
    __##DST_TAG##__ TYPE dst[M * N] = {0}; \
    auto dstIterator = Make##DST_POS##memPtr(dst); \
    auto dstMatrixLayout = Make##DST_FORMAT##Layout<TYPE>(M, N); \
    auto dstTensor = MakeTensor(dstIterator, dstMatrixLayout); \
 \
    __##SRC_TAG##__ TYPE src[M * N] = {0}; \
    auto srcIterator = Make##SRC_POS##memPtr(src); \
    auto srcMatrixLayout = Make##SRC_FORMAT##Layout<TYPE>(M, N); \
    auto srcTensor = MakeTensor(srcIterator, srcMatrixLayout); \
 \
    auto coord = MakeCoord(AscendC::Std::Int<COORD_I>{}, AscendC::Std::Int<COORD_J>{}); \
    constexpr int M_STEP = (sizeof(TYPE) == 1 && TRANSPOSE) ? 2 : 1; \
    constexpr int K_STEP = (sizeof(TYPE) == 4 && TRANSPOSE) ? 2 : 1; \
    MOCKER_CPP(load_cbuf_to_##DST_TAG, void(__ca__ TYPE*, __cbuf__ TYPE*, uint16_t, uint16_t, uint8_t, uint8_t, int16_t, uint16_t, bool)) \
        .times(2) \
        .will(invoke(&load_cbuf_to_##DST_TAG##_stub<TRANSPOSE, TYPE, M_STEP, K_STEP>)); \
    LoadData<LoadDataTraitHolder<TRANSPOSE>::trait>(dstTensor, srcTensor);  \
    LoadData<LoadDataTraitHolder<TRANSPOSE>::trait>(dstTensor, srcTensor, coord);  \
 \
    mockcpp::GlobalMockObject::verify(); \
}


// l1 -> l0A NZ2NZ非转置，覆盖所有TYPE，覆盖传入Coord
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(float, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(float, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 8, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int8_t, 16, 32, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int8_t, 16, 32, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 16, 32, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 16, 32, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Nz, Nz, L1, L0A, cbuf, ca, false, 1, 1);

// ZN2NZ转置，覆盖所有TYPE，覆盖传入Coord
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(float, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(float, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 32, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 32, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 32, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 32, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Zn, Nz, L1, L0A, cbuf, ca, true, 1, 1);

// l1 -> l0B ZN2ZN非转置，覆盖所有TYPE，覆盖传入Coord
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(float, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(float, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int32_t, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int32_t, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 8, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Zn, Zn, L1, L0B, cbuf, cb, false, 1, 1);

// l1 -> l0B NZ2ZN转置，覆盖所有TYPE，覆盖传入Coord
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(bfloat16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(half, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(float, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(float, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int32_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint32_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 32, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int8_t, 32, 32, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 32, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint8_t, 32, 32, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(int16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 0, 0);
TEST_TENSOR_API_LOAD_DATA(uint16_t, 16, 16, Nz, Zn, L1, L0B, cbuf, cb, true, 1, 1);

