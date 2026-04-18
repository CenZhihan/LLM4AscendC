/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <cstring>
#include <numeric>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <cxxabi.h>
#include "mockcpp/mockcpp.hpp"
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"

using namespace AscendC::Te;
using namespace AscendC;

constexpr bool gDebugPrint = false; // Set to true to enable debug printing of captured data

// Mock implementations for data copy about gm2l1 functions
extern void set_loop1_stride_outtol1(uint64_t config);
extern void set_loop2_stride_outtol1(uint64_t config);
extern void set_loop_size_outtol1(uint64_t config);
extern void set_pad_val_outtol1(uint64_t config);
extern void copy_gm_to_cbuf_align_v2(half* dst, half* src, uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                                     uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit,
                                     uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride);
extern void copy_gm_to_cbuf_multi_nd2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
extern void copy_gm_to_cbuf_multi_dn2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
extern void set_mte2_nz_para(uint64_t para);

#define CAPTURE_GM_TO_L1_DEFINITION(type)                                                                              \
    void CaptureCopyGmToCbufAlignV2_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid, uint32_t blockCount,     \
                                           uint32_t blockLen, uint8_t leftPaddingCnt, uint8_t rightPaddingCnt,         \
                                           bool dataSelectBit, uint8_t l2CacheCtl, uint64_t srcStride,                 \
                                           uint32_t dstStride);                                                        \
    void CaptureCopyGmToCbufMultiND2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);           \
    void CaptureCopyGmToCbufMultiDN2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);

CAPTURE_GM_TO_L1_DEFINITION(uint8_t);
CAPTURE_GM_TO_L1_DEFINITION(half);
CAPTURE_GM_TO_L1_DEFINITION(uint16_t);
CAPTURE_GM_TO_L1_DEFINITION(float);
CAPTURE_GM_TO_L1_DEFINITION(uint32_t);

void CaptureSetMTE2NzPara(uint64_t para);

#define MOCKER_GM_TO_L1(type)                                                                                          \
    MOCKER(copy_gm_to_cbuf_align_v2, void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint32_t, uint32_t, uint8_t,      \
                                              uint8_t, bool, uint8_t, uint64_t, uint32_t))                             \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufAlignV2_##type));                                                              \
    MOCKER(copy_gm_to_cbuf_multi_nd2nz,                                                                                \
           void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint64_t, uint8_t, uint16_t, uint32_t, uint64_t, bool))     \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufMultiND2nz_##type));                                                           \
    MOCKER(copy_gm_to_cbuf_multi_dn2nz,                                                                                \
           void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint64_t, uint8_t, uint16_t, uint32_t, uint64_t, bool))     \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufMultiDN2nz_##type))

void ResetCapture();
void PrintCaptureData();
template <typename T>
void PrintTensor(const T& src);

template <typename T, typename U>
void DataCopyGm2L1Sim(const T& dst, const U& src);
template <typename T, typename U, typename Coord>
void DataCopyGm2L1Sim(const T& dst, const U& src, const Coord& coord);

class TensorApiGm2L1 : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    virtual void SetUp()
    {
        ResetCapture();
        MOCKER_GM_TO_L1(uint8_t);
        MOCKER_GM_TO_L1(half);
        MOCKER_GM_TO_L1(uint16_t);
        MOCKER_GM_TO_L1(float);
        MOCKER_GM_TO_L1(uint32_t);
        MOCKER(set_mte2_nz_para, void (*)(uint64_t)).stubs().will(invoke(CaptureSetMTE2NzPara));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
    template <typename T>
    void InitializeData()
    {
        using CastT = Std::conditional_t<sizeof(T) == 1, uint8_t, T>;
        using TT = Std::conditional_t<sizeof(T) == 2, uint16_t, CastT>;
        std::iota(reinterpret_cast<TT*>(src0Gm), reinterpret_cast<TT*>(src0Gm + GmSize), static_cast<TT>(1));
        std::fill(reinterpret_cast<TT*>(l1ABuf), reinterpret_cast<TT*>(l1ABuf + L1Size), static_cast<TT>(1));
        std::fill(reinterpret_cast<TT*>(l1ABufGolden), reinterpret_cast<TT*>(l1ABufGolden + L1Size),
                  static_cast<TT>(1));
    }

private:
    constexpr static uint32_t GmSize = 64 * 1024;
    constexpr static uint32_t L1Size = 64 * 1024;
    __gm__ uint8_t src0Gm[GmSize] = {0};
    __cbuf__ uint8_t l1ABuf[L1Size] = {0};
    __cbuf__ uint8_t l1ABufGolden[L1Size] = {0};
};

#define EXPECT_GM2L1_EQ()                                                                                              \
    bool result = std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden);                                                   \
    EXPECT_TRUE(result);                                                                                               \
    if (gDebugPrint || !result) {                                                                                      \
        PrintCaptureData();                                                                                            \
        PrintTensor(gmA);                                                                                              \
        PrintTensor(l1ATensor);                                                                                        \
        PrintTensor(l1ATensorGolden);                                                                                  \
    }

#define TEST_GM2L1_CONCAT_IMPL_(a, b, c, d) a##_##b##_##c##_##index##_##d
#define TEST_GM2L1_CONCAT_(a, b, c, d) TEST_GM2L1_CONCAT_IMPL_(a, b, c, d)
#define TEST_GM2L1_INNER(type, name, gmALayout, l1ALayout, counter)                                                    \
    TEST_F(TensorApiGm2L1, TEST_GM2L1_CONCAT_(CopyGm2L1Operation, name, type, counter))                                \
    {                                                                                                                  \
        using T = type;                                                                                                \
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);                                  \
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);                            \
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);                \
        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});                                                 \
        InitializeData<T>();                                                                                           \
        atomCopy.Call(l1ATensor, gmA);                                                                                 \
        DataCopyGm2L1Sim(l1ATensorGolden, gmA);                                                                        \
        EXPECT_GM2L1_EQ();                                                                                             \
    }
#define TEST_GM2L1_COORD_INNER(type, name, gmALayout, l1ALayout, makeCoord, counter)                                   \
    TEST_F(TensorApiGm2L1, TEST_GM2L1_CONCAT_(CopyGm2L1OperationWithCoord, name, type, counter))                       \
    {                                                                                                                  \
        using T = type;                                                                                                \
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);                                  \
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);                            \
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);                \
        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});                                                 \
        InitializeData<T>();                                                                                           \
        auto coord = makeCoord;                                                                                        \
        atomCopy.Call(l1ATensor, gmA, coord);                                                                          \
        DataCopyGm2L1Sim(l1ATensorGolden, gmA, coord);                                                                 \
        EXPECT_GM2L1_EQ();                                                                                             \
    }

#define TEST_GM2L1(type, name, gmALayout, l1ALayout) TEST_GM2L1_INNER(type, name, gmALayout, l1ALayout, __COUNTER__)

#define TEST_GM2L1_COORD(type, name, gmALayout, l1ALayout, makeCoord)                                                  \
    TEST_GM2L1_COORD_INNER(type, name, gmALayout, l1ALayout, makeCoord, __COUNTER__)

// ND2ND
TEST_GM2L1(half, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(bfloat16_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(float, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(int8_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(uint8_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(int16_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(uint16_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(int32_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(uint32_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(int64_t, ND2ND1Dim, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(uint64_t, ND2ND1Dim, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(fp8_e4m3fn_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(fp8_e5m2_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))
TEST_GM2L1(hifloat8_t, ND2ND, MakeNDLayout<T>(17, 18), MakeNDLayout<T>(19, 32))

// continuous case
TEST_GM2L1(uint8_t, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint8_t, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 40))
TEST_GM2L1(half, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(half, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 40))
TEST_GM2L1(float, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(float, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 40))
TEST_GM2L1(uint64_t, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint64_t, ND2ND1Dim, MakeNDLayout<T>(1, 17), MakeNDLayout<T>(1, 40))

TEST_GM2L1(uint8_t, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(10, 17))
TEST_GM2L1(uint8_t, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(20, 17))
TEST_GM2L1(half, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(10, 17))
TEST_GM2L1(half, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(20, 17))
TEST_GM2L1(float, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(10, 17))
TEST_GM2L1(float, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(20, 17))
TEST_GM2L1(uint64_t, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(10, 17))
TEST_GM2L1(uint64_t, ND2ND1Dim, MakeNDLayout<T>(10, 17), MakeNDLayout<T>(20, 17))

TEST_GM2L1(half, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(bfloat16_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(float, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(int8_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint8_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(int16_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint16_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(int32_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint32_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(int64_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(uint64_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(fp8_e4m3fn_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(fp8_e5m2_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))
TEST_GM2L1(hifloat8_t, ND2ND1DimInt, MakeNDLayout<T>(Std::Int<1>(), 17), MakeNDLayout<T>(1, 19))

TEST_GM2L1(uint8_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, 1))
TEST_GM2L1(uint8_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, Std::Int<1>()))
TEST_GM2L1(uint16_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, 1))
TEST_GM2L1(uint16_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, Std::Int<1>()))
TEST_GM2L1(float, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, 1))
TEST_GM2L1(float, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, Std::Int<1>()))
TEST_GM2L1(uint64_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, 1))
TEST_GM2L1(uint64_t, ND2ND1DimInt, MakeNDLayout<T>(17, Std::Int<1>()), MakeNDLayout<T>(19, Std::Int<1>()))

// non continuous case, the dst col stride of ND layout needs to be aligned with C0_SIZE
TEST_GM2L1_COORD(uint16_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 16), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint32_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 8), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint8_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint32_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint8_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint32_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint8_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint16_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint32_t, ND2ND, MakeNDLayout<T>(33, 40), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint8_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint16_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint32_t, ND2ND, MakeNDLayout<T>(33, 25), MakeNDLayout<T>(19, 32), MakeCoord(16, 16))

// ND2Nz
TEST_GM2L1(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(18, 18), MakeNzLayout<T>(19, 20))
TEST_GM2L1(fp4x2_e1m2_t, ND2Nz, MakeNDLayout<T>(18, 18), MakeNzLayout<T>(19, 20))
TEST_GM2L1(fp8_e4m3fn_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(fp8_e5m2_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(hifloat8_t, ND2Nz, MakeNDLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(half, ND2Nz, MakeNDLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(bfloat16_t, ND2Nz, MakeNDLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(float, ND2Nz, MakeNDLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(int8_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint8_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int16_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint16_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int32_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint32_t, ND2Nz, MakeNDLayout<T>(18, 9), MakeNzLayout<T>(19, 10))

TEST_GM2L1(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(68, 68), MakeNzLayout<T>(69, 70))
TEST_GM2L1(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(18, 18), MakeNzLayout<T>(69, 70))

// fp4 col % 2 must be 0
TEST_GM2L1_COORD(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(33, 26), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint16_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint32_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(33, 26), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint8_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint32_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(33, 26), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint8_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint16_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint32_t, ND2Nz, MakeNDLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(fp4x2_e2m1_t, ND2Nz, MakeNDLayout<T>(64, 64), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint8_t, ND2Nz, MakeNDLayout<T>(64, 64), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint16_t, ND2Nz, MakeNDLayout<T>(64, 64), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))
TEST_GM2L1_COORD(uint32_t, ND2Nz, MakeNDLayout<T>(64, 64), MakeNzLayout<T>(19, 18), MakeCoord(16, 16))

// ND2Zn
TEST_GM2L1(fp8_e4m3fn_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(fp8_e5m2_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(hifloat8_t, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(bfloat16_t, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(float, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(int8_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint8_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(int16_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint16_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(int32_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint32_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))

TEST_GM2L1(uint8_t, ND2Zn, MakeNDLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 38))
TEST_GM2L1(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(39, 18))
TEST_GM2L1(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(39, 48))
TEST_GM2L1(uint32_t, ND2Zn, MakeNDLayout<T>(16, 32), MakeZnLayout<T>(16, 32))
TEST_GM2L1(uint32_t, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(19, 18))

TEST_GM2L1_COORD(half, ND2Zn, MakeNDLayout<T>(18, 17), MakeZnLayout<T>(39, 48), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint16_t, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(float, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(float, ND2Zn, MakeNDLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))

// DN2Nz
TEST_GM2L1(fp8_e4m3fn_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(fp8_e5m2_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(hifloat8_t, DN2Nz, MakeDNLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(half, DN2Nz, MakeDNLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(bfloat16_t, DN2Nz, MakeDNLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(float, DN2Nz, MakeDNLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(int8_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint8_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int16_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint16_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int32_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint32_t, DN2Nz, MakeDNLayout<T>(18, 9), MakeNzLayout<T>(19, 10))

TEST_GM2L1(uint8_t, DN2Nz, MakeDNLayout<T>(16, 32), MakeNzLayout<T>(16, 32))
TEST_GM2L1(uint8_t, DN2Nz, MakeDNLayout<T>(17, 18), MakeNzLayout<T>(18, 19))
TEST_GM2L1(uint16_t, DN2Nz, MakeDNLayout<T>(16, 32), MakeNzLayout<T>(16, 32))
TEST_GM2L1(uint16_t, DN2Nz, MakeDNLayout<T>(17, 18), MakeNzLayout<T>(18, 19))
TEST_GM2L1(float, DN2Nz, MakeDNLayout<T>(16, 32), MakeNzLayout<T>(16, 32))
TEST_GM2L1(float, DN2Nz, MakeDNLayout<T>(17, 18), MakeNzLayout<T>(18, 19))

TEST_GM2L1_COORD(half, DN2Nz, MakeDNLayout<T>(18, 17), MakeNzLayout<T>(39, 48), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint16_t, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(float, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(float, DN2Nz, MakeDNLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))

// DN2Zn
TEST_GM2L1(fp4x2_e1m2_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(20, 10))
TEST_GM2L1(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(20, 10))
TEST_GM2L1(fp8_e4m3fn_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(fp8_e5m2_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(hifloat8_t, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(half, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(bfloat16_t, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(float, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(19, 18))
TEST_GM2L1(int8_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint8_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(int16_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint16_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(int32_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint32_t, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))

TEST_GM2L1(float, DN2Zn, MakeDNLayout<T>(16, 32), MakeZnLayout<T>(16, 32))
TEST_GM2L1(float, DN2Zn, MakeDNLayout<T>(18, 9), MakeZnLayout<T>(19, 10))
TEST_GM2L1(uint16_t, DN2Zn, MakeDNLayout<T>(18, 18), MakeZnLayout<T>(19, 20))
TEST_GM2L1(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(18, 18), MakeZnLayout<T>(19, 20))
TEST_GM2L1(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(68, 68), MakeZnLayout<T>(69, 70))

TEST_GM2L1_COORD(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(40, 48), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(34, 25), MakeZnLayout<T>(20, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp4x2_e2m1_t, DN2Zn, MakeDNLayout<T>(34, 25), MakeZnLayout<T>(20, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(half, DN2Zn, MakeDNLayout<T>(18, 17), MakeZnLayout<T>(39, 48), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint16_t, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(float, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(float, DN2Zn, MakeDNLayout<T>(33, 25), MakeZnLayout<T>(19, 18), MakeCoord(10, 10))

// Nz2Nz
TEST_GM2L1(fp8_e4m3fn_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(fp8_e5m2_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(hifloat8_t, Nz2Nz, MakeNzLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(half, Nz2Nz, MakeNzLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(bfloat16_t, Nz2Nz, MakeNzLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(float, Nz2Nz, MakeNzLayout<T>(18, 17), MakeNzLayout<T>(19, 18))
TEST_GM2L1(int8_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint8_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int16_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint16_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int32_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint32_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(int64_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))
TEST_GM2L1(uint64_t, Nz2Nz, MakeNzLayout<T>(18, 9), MakeNzLayout<T>(19, 10))

TEST_GM2L1(uint8_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(38, 40))
TEST_GM2L1(uint8_t, Nz2Nz, MakeNzLayout<T>(13, 18), MakeNzLayout<T>(14, 20))
TEST_GM2L1(uint8_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(19, 20))
TEST_GM2L1(uint16_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(38, 40))
TEST_GM2L1(uint16_t, Nz2Nz, MakeNzLayout<T>(13, 18), MakeNzLayout<T>(14, 20))
TEST_GM2L1(uint16_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(19, 20))
TEST_GM2L1(uint32_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(38, 40))
TEST_GM2L1(uint32_t, Nz2Nz, MakeNzLayout<T>(13, 18), MakeNzLayout<T>(14, 20))
TEST_GM2L1(uint32_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(19, 20))
TEST_GM2L1(uint64_t, Nz2Nz, MakeNzLayout<T>(13, 18), MakeNzLayout<T>(14, 20))
TEST_GM2L1(uint64_t, Nz2Nz, MakeNzLayout<T>(17, 18), MakeNzLayout<T>(19, 20))

TEST_GM2L1_COORD(half, Nz2Nz, MakeNzLayout<T>(18, 17), MakeNzLayout<T>(39, 48), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint8_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint16_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint16_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(float, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(float, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))
TEST_GM2L1_COORD(uint64_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(0, 0))
TEST_GM2L1_COORD(uint64_t, Nz2Nz, MakeNzLayout<T>(33, 25), MakeNzLayout<T>(19, 18), MakeCoord(10, 10))

// ScaleA
// scaleA col direction, col % 2 must be 0
TEST_GM2L1(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(18, 34), MakeZzLayout<T>(19, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(18, 34), MakeZzLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(36, 34), MakeZzLayout<T>(40, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(32, 32), MakeZzLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(2, 8), MakeZzLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(36, 34), MakeZzLayout<T>(40, 36), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(36, 34), MakeZzLayout<T>(70, 70), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAND2Zz, MakeScaleANDLayout<T>(32, 32), MakeZzLayout<T>(32, 32), MakeCoord(10, 10))

TEST_GM2L1(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(18, 34), MakeZzLayout<T>(19, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(18, 34), MakeZzLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(36, 34), MakeZzLayout<T>(40, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(32, 32), MakeZzLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(2, 8), MakeZzLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(36, 34), MakeZzLayout<T>(40, 36), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(36, 34), MakeZzLayout<T>(70, 70), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleADN2Zz, MakeScaleADNLayout<T>(32, 32), MakeZzLayout<T>(32, 32), MakeCoord(10, 10))

TEST_GM2L1(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(18, 34), MakeZzLayout<T>(19, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(18, 34), MakeZzLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(36, 34), MakeZzLayout<T>(40, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(32, 32), MakeZzLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(2, 8), MakeZzLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(36, 34), MakeZzLayout<T>(26, 26), MakeCoord(16, 2))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(36, 34), MakeZzLayout<T>(40, 36), MakeCoord(16, 2))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(36, 34), MakeZzLayout<T>(70, 70), MakeCoord(16, 2))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleAZz2Zz, MakeZzLayout<T>(32, 32), MakeZzLayout<T>(32, 32), MakeCoord(16, 2))

// ScaleB
// scaleB row direction, row % 2 must be 0
TEST_GM2L1(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(18, 34), MakeNnLayout<T>(20, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(18, 34), MakeNnLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(36, 33), MakeNnLayout<T>(40, 35))
TEST_GM2L1(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(32, 32), MakeNnLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(2, 8), MakeNnLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(36, 33), MakeNnLayout<T>(40, 35), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(36, 33), MakeNnLayout<T>(70, 70), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBND2Nn, MakeScaleBNDLayout<T>(32, 32), MakeNnLayout<T>(32, 32), MakeCoord(10, 10))

TEST_GM2L1(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(18, 34), MakeNnLayout<T>(20, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(18, 34), MakeNnLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(36, 33), MakeNnLayout<T>(40, 35))
TEST_GM2L1(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(32, 32), MakeNnLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(2, 8), MakeNnLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(36, 33), MakeNnLayout<T>(40, 35), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(36, 33), MakeNnLayout<T>(70, 70), MakeCoord(10, 10))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBDN2Nn, MakeScaleBDNLayout<T>(32, 32), MakeNnLayout<T>(32, 32), MakeCoord(10, 10))

TEST_GM2L1(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(18, 34), MakeNnLayout<T>(20, 36))
TEST_GM2L1(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(18, 34), MakeNnLayout<T>(40, 70))
TEST_GM2L1(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(36, 33), MakeNnLayout<T>(40, 35))
TEST_GM2L1(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(32, 32), MakeNnLayout<T>(32, 32))
TEST_GM2L1(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(2, 8), MakeNnLayout<T>(4, 12))

TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(0, 0))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(36, 33), MakeNnLayout<T>(26, 25), MakeCoord(2, 16))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(36, 33), MakeNnLayout<T>(40, 35), MakeCoord(2, 16))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(36, 33), MakeNnLayout<T>(70, 70), MakeCoord(2, 16))
TEST_GM2L1_COORD(fp8_e8m0_t, ScaleBNnNn, MakeNnLayout<T>(32, 32), MakeNnLayout<T>(32, 32), MakeCoord(2, 16))

// PrintTensor
template <typename T>
void PrintTensor(const T& src)
{
    using srcType = typename T::elementType;
    auto srcLayout = src.Layout();
    uint32_t M0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout);
    uint32_t N0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    uint32_t M1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t N1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    if constexpr (Std::is_same_v<srcType, fp8_e8m0_t> && IsScaleANDFormat<T>::value) {
        std::cout << "ScaleAND";
    } else if constexpr (Std::is_same_v<srcType, fp8_e8m0_t> && IsScaleADNFormat<T>::value) {
        std::cout << "ScaleADN";
    } else if constexpr (Std::is_same_v<srcType, fp8_e8m0_t> && IsScaleANDFormat<T>::value) {
        std::cout << "ScaleAND";
    } else if constexpr (Std::is_same_v<srcType, fp8_e8m0_t> && IsScaleBDNFormat<T>::value) {
        std::cout << "ScaleBDN";
    } else if constexpr (IsNDFormat<T>::value) {
        std::cout << "ND";
    } else if constexpr (IsDNFormat<T>::value) {
        std::cout << "DN";
    } else if constexpr (IsNZFormat<T>::value) {
        std::cout << "NZ";
    } else if constexpr (IsZNFormat<T>::value) {
        std::cout << "ZN";
    } else if constexpr (IsZZFormat<T>::value) {
        std::cout << "ZZ";
    } else if constexpr (IsNNFormat<T>::value) {
        std::cout << "NN";
    } else {
        std::cout << "UnknownLayout";
    }
    if (M0 == 1 && N0 == 1) { // for 2D layout, print in 2D format
        std::cout << " Layout Result (2D) (" << M1 << ", " << N1 << "): " << std::endl;
        for (int i = 0; i < M1; i++) {
            std::cout << i << ":\t";
            for (int j = 0; j < N1; j++) {
                auto dataAddr = &(src[MakeCoord(i, j)]);
                if constexpr (sizeof(srcType) == 1) {
                    std::cout << static_cast<uint32_t>(*(reinterpret_cast<uint8_t*>(dataAddr))) << "\t";
                } else if constexpr (Std::is_same_v<srcType, half>) {
                    std::cout << static_cast<float>(*dataAddr) << "\t";
                } else {
                    std::cout << *dataAddr << "\t";
                }
            }
            std::cout << std::endl;
        }
    } else { // for NZ, ZN, ZZ, print in 4D format
        std::cout << " Layout Result (4D) (" << M1 << ", " << N1 << ", " << M0 << ", " << N0 << "): " << std::endl;
        for (int i0 = 0; i0 < M1; i0++) {
            for (int i1 = 0; i1 < M0; i1++) {
                for (int j0 = 0; j0 < N1; j0++) {
                    uint32_t block_id = j0 * M1 + i0;
                    for (int j1 = 0; j1 < N0; j1++) {
                        auto dataAddr = &(src[MakeCoord(MakeCoord(i1, i0), MakeCoord(j1, j0))]);
                        if constexpr (sizeof(srcType) == 1) {
                            std::cout << static_cast<uint32_t>(*(reinterpret_cast<uint8_t*>(dataAddr))) << "\t";
                        } else if constexpr (Std::is_same_v<srcType, half>) {
                            std::cout << static_cast<float>(*dataAddr) << "\t";
                        } else {
                            std::cout << *dataAddr << "\t";
                        }
                    }
                    std::cout << "|";
                }
                std::cout << std::endl;
            }
            std::cout << "-----------------------------------------" << std::endl;
        }
    }
}

inline void __print_type_hierarchy(const std::string& type_str)
{
    int indent_level = 1;
    const int indent_spaces = 4; // 每层缩进的空格数
    for (int s = 0; s < indent_level * indent_spaces; ++s)
        std::cout << " ";
    for (size_t i = 0; i < type_str.size(); ++i) {
        char c = type_str[i];
        if (c == '<') {
            // 遇到 <，换行并增加缩进
            std::cout << c << "\n";
            indent_level++;
            // 打印缩进
            for (int s = 0; s < indent_level * indent_spaces; ++s)
                std::cout << " ";
        } else if (c == ',' && indent_level > 0) {
            // 遇到逗号，换行并保持当前缩进
            std::cout << c << "\n";
            for (int s = 0; s < indent_level * indent_spaces - 1; ++s)
                std::cout << " ";
        } else if (c == '>') {
            // 遇到 >，先换行，减少缩进，再打印 >
            std::cout << "\n";
            indent_level--;
            for (int s = 0; s < indent_level * indent_spaces; ++s)
                std::cout << " ";
            std::cout << c;
        } else {
            // 普通字符直接打印
            std::cout << c;
        }
    }
    std::cout << std::endl;
}

template <typename T, typename... Args>
inline void PrintTypeHierarchy(const Args&... args)
{
    if constexpr (!std::is_same_v<T, void>) {
        std::cout << "Type Hierarchy for: ";
    }
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
    if constexpr (std::is_same_v<T, void>) {
        return;
    }
    std::string raw_name = typeid(T).name();
    int status = -4;
    char* res = abi::__cxa_demangle(raw_name.c_str(), NULL, NULL, &status);
    std::string ret = (status == 0) ? res : raw_name;
    if (status == 0)
        std::free(res);
    __print_type_hierarchy(ret);
}

// Sim gm2l1 copy by cpu
template <typename T, typename U>
void SimND2ND(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    uint32_t M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
    auto dstRowStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    if (M == 1 || N == 1 || (N == N1 && srcRowStride == N && dstRowStride == N1)) {
        // if src is already in row major or column major format, treat it as M=1 or N=1 to simplify the copy
        uint32_t dataLen = M * N;
        uint32_t alignN = (dataLen + c0Elements - 1) / c0Elements * c0Elements;
        for (uint32_t i = 0; i < alignN; i++) {
            if (i < dataLen) {
                dst.Data()[i] = src.Data()[i];
            } else {
                // padding with 0 if out of bound
                dst.Data()[i] = static_cast<srcType>(0);
            }
        }
        return;
    }

    N1 = (N1 + c0Elements - 1) / c0Elements * c0Elements; // align N1 to C0 boundary

    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow * srcRowStride + srcCol * srcColStride];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // padding with 0 if out of bound
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimND2Nz(const T& dst, const U& src)
{
    static_assert(IsNDFormat<U>::value && IsNZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcSN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t srcIndex = srcRow * srcSM1 + srcCol * srcSN1;
                    uint32_t dstIndex = ((n1 * M1 + m1) * M0 + m0) * N0 + n0;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimND2Zn(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t n0 = 0; n0 < N0; n0++) {
                for (uint32_t m0 = 0; m0 < M0; m0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * N0 + n0) * M0 + m0;

                    if (srcCol < N && srcRow < M) {
                        dst.Data()[dstIndex] = src.Data()[srcCol + srcRow * srcRowStride];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimDN2Nz(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(!is_b4_type<srcType>, "DN2NZ does not support b4 type");
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((n1 * M1 + m1) * M0 + m0) * N0 + n0;
                    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow + srcCol * srcColStride];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimDN2Zn(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t n0 = 0; n0 < N0; n0++) {
                for (uint32_t m0 = 0; m0 < M0; m0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * N0 + n0) * M0 + m0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcCol < N && srcRow < M) {
                        dst.Data()[dstIndex] = src.Data()[srcCol * srcColStride + srcRow];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimNz2Nz(const T& dst, const U& src)
{
    static_assert(IsNZFormat<U>::value && IsNZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto srcM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto srcN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcSN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 0>(srcLayout);
    auto srcSN0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 0>(srcLayout);

    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    auto dstSM1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
    auto dstSN1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
    auto dstSM0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 0>(dstLayout);
    auto dstSN0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 0>(dstLayout);

    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcIndex = n1 * srcSN1 + m1 * srcSM1 + n0 * srcSN0 + m0 * srcSM0;
                    uint32_t dstIndex = n1 * dstSN1 + m1 * dstSM1 + n0 * dstSN0 + m0 * dstSM0;
                    if (m1 < srcM1 && n1 < srcN1) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    }
                    // no pad
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleAND2Zz(const T& dst, const U& src)
{
    // static_assert(IsScaleANDFormat<U>::value && IsZZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow * srcRowStride + srcCol];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        // use dn2nz way to pad, which means padding in the raw row direction, actual col direction
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleADN2Zz(const T& dst, const U& src)
{
    static_assert(IsScaleADNFormat<U>::value && IsZZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto SN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    auto BN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto N = SN * BN;

    auto srcBColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[n1 * srcBColStride + srcRow * 2 + n0];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        // use dn2nz way to pad, which means padding in the raw row direction, actual col direction
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleAZz2Zz(const T& dst, const U& src)
{
    static_assert(IsZZFormat<U>::value && IsZZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto srcM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto srcN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcSN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 0>(srcLayout);
    auto srcSN0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 0>(srcLayout);

    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcIndex = m1 * srcSM1 + n1 * srcSN1 + m0 * srcSM0 + n0 * srcSN0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    if (m1 < srcM1 && n1 < srcN1) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    }
                    // no pad
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleBND2Nn(const T& dst, const U& src)
{
    static_assert(IsScaleBNDFormat<U>::value && IsNNFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto SM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout);
    auto BM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto SN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    auto BN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto srcSM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcSN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    uint32_t c0Elements = C0_ELEMENT<half>; // sim by b16
    uint32_t srcColNAlignC0 = ((SN * BN + c0Elements - 1) / c0Elements) * c0Elements;
    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t n0 = 0; n0 < N0; n0++) {
                for (uint32_t m0 = 0; m0 < M0; m0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    // M1 N1 N0 M0
                    uint32_t srcIndex = m1 * srcSM1 + srcCol * srcSN1 + m0;
                    // N1 M1 N0 M0
                    uint32_t dstIndex = ((n1 * M1 + m1) * N0 + n0) * M0 + m0;
                    if (srcRow < SM * BM && srcCol < SN * BN) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    } else if (srcRow < SM * BM && srcCol >= SN * BN && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next B*N boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleBDN2Nn(const T& dst, const U& src)
{
    static_assert(IsScaleBDNFormat<U>::value && IsNNFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    uint32_t SM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout);
    uint32_t BM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t SN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    uint32_t BN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    uint32_t srcSM1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t srcSN1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    uint32_t c0Elements = C0_ELEMENT<half>; // sim by b16
    uint32_t srcColNAlignC0 = ((SN * BN + c0Elements - 1) / c0Elements) * c0Elements;
    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    // M1 N1 M0 N0
                    uint32_t srcIndex = srcCol * srcSN1 + srcRow * srcSM1;
                    // N1 M1 N0 M0
                    uint32_t dstIndex = ((n1 * M1 + m1) * N0 + n0) * M0 + m0;
                    if (srcRow < SM * BM && srcCol < SN * BN) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    } else if (srcRow < SM * BM && srcCol >= SN * BN && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next B*N boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleBNn2Nn(const T& dst, const U& src)
{
    static_assert(IsNNFormat<U>::value && IsNNFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    uint32_t SM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout);
    uint32_t BM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t SN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    uint32_t BN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    uint32_t srcSM = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 0>(srcLayout);
    uint32_t srcSN = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 0>(srcLayout);
    uint32_t srcBM = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t srcBN = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    uint32_t c0Elements = C0_ELEMENT<half>; // sim by b16
    uint32_t srcColNAlignC0 = ((SN * BN + c0Elements - 1) / c0Elements) * c0Elements;

    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t srcIndex = n1 * srcBN + m1 * srcBM + n0 * srcSN + m0 * srcSM;
                    uint32_t dstIndex = ((n1 * M1 + m1) * N0 + n0) * M0 + m0;
                    if (srcRow < SM * BM && srcCol < SN * BN) {
                        dst.Data()[dstIndex] = src.Data()[srcIndex];
                    } else if (srcRow < SM * BM && srcCol >= SN * BN && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next B*N boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void DataCopyGm2L1Sim(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
        SimND2ND(dst, src);
    } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
        SimND2Nz(dst, src);
    } else if constexpr (IsNDFormat<U>::value && IsZNFormat<T>::value) {
        SimND2Zn(dst, src);
    } else if constexpr (IsDNFormat<U>::value && IsNZFormat<T>::value) {
        SimDN2Nz(dst, src);
    } else if constexpr (IsDNFormat<U>::value && IsZNFormat<T>::value) {
        SimDN2Zn(dst, src);
    } else if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
        SimNz2Nz(dst, src);
    } else if constexpr (IsScaleANDFormat<U>::value && IsZZFormat<T>::value) {
        SimScaleAND2Zz(dst, src);
    } else if constexpr (IsScaleADNFormat<U>::value && IsZZFormat<T>::value) {
        SimScaleADN2Zz(dst, src);
    } else if constexpr (IsZZFormat<U>::value && IsZZFormat<T>::value) {
        SimScaleAZz2Zz(dst, src);
    } else if constexpr (IsScaleBNDFormat<U>::value && IsNNFormat<T>::value) {
        SimScaleBND2Nn(dst, src);
    } else if constexpr (IsScaleBDNFormat<U>::value && IsNNFormat<T>::value) {
        SimScaleBDN2Nn(dst, src);
    } else if constexpr (IsNNFormat<U>::value && IsNNFormat<T>::value) {
        SimScaleBNn2Nn(dst, src);
    } else {
        // assert error
        static_assert(Std::is_same_v<T, U>, "The data format is not supported.");
    }
}

template <typename T, typename U, typename Coord>
void DataCopyGm2L1Sim(const T& dst, const U& src, const Coord& coord)
{
    auto sliceTensor = src(coord, dst);
    DataCopyGm2L1Sim(dst, sliceTensor);
}

// Simulate hardware instruction.
struct CopyGm2L1AlignV2Capture {
    void* dst = nullptr;
    void* src = nullptr;
    uint32_t blockCount = 0;
    uint32_t blockLen = 0;
    uint8_t leftPaddingCnt = 0;
    uint8_t rightPaddingCnt = 0;
    bool dataSelectBit = false;
    uint8_t l2CacheCtl = 0;
    uint64_t srcStride = 0;
    uint32_t dstStride = 0;
};

struct CopyGm2L1ND2NzCapture {
    void* dst = nullptr;
    void* src = nullptr;
    uint64_t loop1SrcStride = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t loop4SrcStride = 0;
    bool enableSmallC0 = false;
};

struct CopyGm2L1DN2NzCapture {
    void* dst = nullptr;
    void* src = nullptr;
    uint64_t loop1SrcStride = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t loop4SrcStride = 0;
    bool enableSmallC0 = false;
};

struct CopyGm2L1NzParaCapture {
    union {
        struct {
            uint16_t ndNum;          // MTE2_NZ_PARA[15:0]
            uint16_t loop2DstStride; // MTE2_NZ_PARA[31:16]
            uint16_t loop3DstStride; // MTE2_NZ_PARA[47:32]
            uint16_t loop4DstStride; // MTE2_NZ_PARA[63:48]
        };
        uint64_t mte2NzPara;
    };
};

// Global capture object
std::vector<CopyGm2L1AlignV2Capture> gGm2L1AlignV2Captures;
std::vector<CopyGm2L1ND2NzCapture> gGm2L1ND2NzCaptures;
std::vector<CopyGm2L1DN2NzCapture> gGm2L1DN2NzCaptures;
std::vector<CopyGm2L1NzParaCapture> gGm2L1NzParaCaptures;

// Reset capture data
void ResetCapture()
{
    gGm2L1AlignV2Captures.clear();
    gGm2L1ND2NzCaptures.clear();
    gGm2L1DN2NzCaptures.clear();
    gGm2L1NzParaCaptures.clear();
}

void PrintCaptureData()
{
    for (const auto& capture : gGm2L1AlignV2Captures) {
        std::cout << "CopyGmToCbufAlignV2 Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", blockCount: " << capture.blockCount << ", blockLen: " << capture.blockLen
                  << ", leftPaddingCnt: " << static_cast<int>(capture.leftPaddingCnt)
                  << ", rightPaddingCnt: " << static_cast<int>(capture.rightPaddingCnt)
                  << ", l2CacheCtl: " << static_cast<int>(capture.l2CacheCtl) << ", srcStride: " << capture.srcStride
                  << ", dstStride: " << capture.dstStride << std::endl;
    }

    for (const auto& capture : gGm2L1ND2NzCaptures) {
        std::cout << "CopyGmToCbufMultiND2nz Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", loop1SrcStride: " << capture.loop1SrcStride << ", nValue: " << capture.nValue
                  << ", dValue: " << capture.dValue << ", loop4SrcStride: " << capture.loop4SrcStride
                  << ", enableSmallC0: " << std::boolalpha << capture.enableSmallC0 << std::endl;
    }

    for (const auto& capture : gGm2L1DN2NzCaptures) {
        std::cout << "CopyGmToCbufMultiDN2nz Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", loop1SrcStride: " << capture.loop1SrcStride << ", nValue: " << capture.nValue
                  << ", dValue: " << capture.dValue << ", loop4SrcStride: " << capture.loop4SrcStride
                  << ", enableSmallC0: " << std::boolalpha << capture.enableSmallC0 << std::endl;
    }

    for (const auto& capture : gGm2L1NzParaCaptures) {
        std::cout << "SetMTE2NzPara Capture - mte2NzPara: " << capture.mte2NzPara << ", ndNum: " << capture.ndNum
                  << ", loop2DstStride: " << capture.loop2DstStride << ", loop3DstStride: " << capture.loop3DstStride
                  << ", loop4DstStride: " << capture.loop4DstStride << std::endl;
    }
}

extern void copy_gm_to_cbuf_multi_nd2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
template <typename T>
void SimulateND2nzDataCopy(T* dst, T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
                           uint64_t loop4SrcStride, bool enableSmallC0)
{
    if (gGm2L1NzParaCaptures.empty()) {
        return;
    }
    uint16_t ndNum = gGm2L1NzParaCaptures.back().ndNum;
    uint16_t loop2DstStride = gGm2L1NzParaCaptures.back().loop2DstStride;
    uint16_t loop3DstStride = gGm2L1NzParaCaptures.back().loop3DstStride;
    uint16_t loop4DstStride = gGm2L1NzParaCaptures.back().loop4DstStride;
    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Elements = C0_SIZE<T> / typeSize; // Number of elements in one C0 block
    if (enableSmallC0) {
        for (int h = 0; h < ndNum; h++) {
            const uint8_t* srcNDAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstNDAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;

            uint16_t nCeil = (nValue + 3) / 4;
            for (int j = 0; j < nCeil; j++) {
                const uint8_t* srcNAddr = (j < nValue) ? (srcNDAddr + j * loop1SrcStride) : nullptr;
                uint8_t* dstNAddr = dstNDAddr + j * 4 * typeSize;
                for (int k = 0; k < 4; k++) {
                    uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                    if ((k < dValue) && (srcNAddr != nullptr)) {
                        const uint8_t* srcEleAddr = srcNAddr + k * typeSize;
                        std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                    } else {
                        std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                    }
                }
            }
        }
    } else {
        uint32_t blockNum = (dValue + c0Elements - 1) / c0Elements;
        for (int h = 0; h < ndNum; h++) {
            const uint8_t* srcNDAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstNDAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;
            for (int i = 0; i < blockNum; i++) {
                const uint8_t* srcBlockAddr = srcNDAddr + i * C0_SIZE<T>;
                uint8_t* dstBlockAddr = dstNDAddr + i * loop3DstStride * C0_SIZE<T>;

                for (int j = 0; j < nValue; j++) {
                    const uint8_t* srcNAddr = srcBlockAddr + j * loop1SrcStride;
                    uint8_t* dstNAddr = dstBlockAddr + j * loop2DstStride * C0_SIZE<T>;
                    for (int k = 0; k < c0Elements; k++) {
                        uint32_t srcEleIndex = i * c0Elements + k;
                        uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                        if (srcEleIndex < dValue) {
                            const uint8_t* srcEleAddr = srcNAddr + k * typeSize;
                            std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                        } else {
                            std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                        }
                    }
                }
            }
        }
    }
}

extern void copy_gm_to_cbuf_multi_dn2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
template <typename T>
void SimulateDN2nzDataCopy(T* dst, T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
                           uint64_t loop4SrcStride, bool enableSmallC0)
{
    if (gGm2L1NzParaCaptures.empty()) {
        return;
    }
    uint16_t dnNum = gGm2L1NzParaCaptures.back().ndNum;
    uint16_t loop2DstStride = gGm2L1NzParaCaptures.back().loop2DstStride;
    uint16_t loop3DstStride = gGm2L1NzParaCaptures.back().loop3DstStride;
    uint16_t loop4DstStride = gGm2L1NzParaCaptures.back().loop4DstStride;
    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Elements = C0_SIZE<T> / typeSize; // Number of elements in one C0 block
    if (enableSmallC0) {
        for (int h = 0; h < dnNum; h++) {
            const uint8_t* srcDNAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstDNAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;

            uint16_t nCeil = (nValue + 3) / 4;
            for (int j = 0; j < nCeil; j++) {
                const uint8_t* srcNAddr = (j < nValue) ? (srcDNAddr + j * typeSize) : nullptr;
                uint8_t* dstNAddr = dstDNAddr + j * 4 * typeSize;
                for (int k = 0; k < 4; k++) {
                    uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                    if ((k < dValue) && (srcNAddr != nullptr)) {
                        const uint8_t* srcEleAddr = srcNAddr + k * loop1SrcStride;
                        std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                    } else {
                        std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                    }
                }
            }
        }
    } else {
        uint32_t blockNum = (dValue + c0Elements - 1) / c0Elements;
        for (int h = 0; h < dnNum; h++) {
            const uint8_t* srcDNAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstDNAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;
            for (int i = 0; i < blockNum; i++) {
                const uint8_t* srcBlockAddr = srcDNAddr + i * loop1SrcStride * c0Elements;
                uint8_t* dstBlockAddr = dstDNAddr + i * loop3DstStride * C0_SIZE<T>;

                for (int j = 0; j < nValue; j++) {
                    const uint8_t* srcNAddr = srcBlockAddr + j * typeSize;
                    uint8_t* dstNAddr = dstBlockAddr + j * loop2DstStride * C0_SIZE<T>;
                    for (int k = 0; k < c0Elements; k++) {
                        uint32_t srcEleIndex = i * c0Elements + k;
                        uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                        if (srcEleIndex < dValue) {
                            const uint8_t* srcEleAddr = srcNAddr + k * loop1SrcStride;
                            std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                        } else {
                            std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                        }
                    }
                }
            }
        }
    }
}

extern void copy_gm_to_cbuf_align_v2(half* dst, half* src, uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                                     uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit,
                                     uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride);
template <typename T>
void SimulateAlignV2DataCopy(T* dst, T* src, uint32_t blockCount, uint32_t blockLen, uint8_t leftPaddingCnt,
                             uint8_t rightPaddingCnt, bool dataSelectBit, uint64_t srcStride, uint32_t dstStride)
{
    bool isLPRPMode = (leftPaddingCnt > 0) || (rightPaddingCnt > 0);
    bool isCompactMode = (dstStride == blockLen);
    uint32_t totalBurstSize = blockLen + leftPaddingCnt * sizeof(T) + rightPaddingCnt * sizeof(T);
    uint32_t padSize = (totalBurstSize % C0_SIZE<T> == 0) ? 0 : (C0_SIZE<T> - (totalBurstSize % C0_SIZE<T>));
    uint32_t padElem = padSize / sizeof(T);
    // compact mode, left and right pad cnt is zero, dstStride equals blockLen, can directly copy without padding
    if (isLPRPMode) {
        // In LPRP mode, dstStride should be aligned to C0 size
        EXPECT_TRUE(dstStride % C0_SIZE<T> == 0);
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            uint8_t* srcBurst = reinterpret_cast<uint8_t*>(src) + blockId * srcStride;
            uint8_t* dstBurst = reinterpret_cast<uint8_t*>(dst) + blockId * dstStride;

            if (leftPaddingCnt > 0) {
                std::fill(dstBurst, dstBurst + leftPaddingCnt * sizeof(T), 0); // Padding with zeros
            }
            std::copy(srcBurst, srcBurst + blockLen, dstBurst + leftPaddingCnt * sizeof(T));

            uint32_t rightPadOffset = leftPaddingCnt * sizeof(T) + blockLen;
            if (rightPaddingCnt > 0) {
                std::fill(dstBurst + rightPadOffset, dstBurst + rightPadOffset + rightPaddingCnt * sizeof(T),
                          0); // Padding with zeros
            }

            uint32_t padOffset = leftPaddingCnt * sizeof(T) + blockLen + rightPaddingCnt * sizeof(T);
            if (padElem > 0) {
                std::fill(dstBurst + padOffset, dstBurst + padOffset + padElem * sizeof(T), 0); // Padding with zeros
            }
        }
        return;
    }
    if (isCompactMode) {
        uint8_t* srcBase = reinterpret_cast<uint8_t*>(src);
        uint8_t* dstBase = reinterpret_cast<uint8_t*>(dst);
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            const uint8_t* srcBurst = srcBase + blockId * srcStride;
            uint8_t* dstBurst = dstBase + blockId * dstStride;
            std::copy(srcBurst, srcBurst + blockLen, dstBurst);
        }
        // check tail padding
        uint32_t totalDataLen = blockCount * blockLen;
        uint64_t aligndSize = ((totalDataLen + C0_SIZE<T> - 1) / C0_SIZE<T>)*C0_SIZE<T>;
        if (aligndSize > totalDataLen) {
            uint8_t* padStart = dstBase + totalDataLen;
            std::fill(padStart, padStart + (aligndSize - totalDataLen), 0); // Padding with zeros
        }
    } else {
        // normal mode
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            uint8_t* srcBurst = reinterpret_cast<uint8_t*>(src) + blockId * srcStride;
            uint8_t* dstBurst = reinterpret_cast<uint8_t*>(dst) + blockId * dstStride;
            std::copy(srcBurst, srcBurst + blockLen, dstBurst);
            if (padElem > 0) {
                uint8_t* padStart = dstBurst + blockLen;
                std::fill(padStart, padStart + padElem * sizeof(T), 0); // Padding with zeros
            }
        }
    }
}

extern void set_mte2_nz_para(uint64_t para);
void CaptureSetMTE2NzPara(uint64_t para)
{
    CopyGm2L1NzParaCapture capture;
    capture.mte2NzPara = para;
    gGm2L1NzParaCaptures.push_back(capture);
}

#define CAPTURE_GM_TO_L1_IMPL(type)                                                                                    \
    void CaptureCopyGmToCbufAlignV2_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid, uint32_t blockCount,     \
                                           uint32_t blockLen, uint8_t leftPaddingCnt, uint8_t rightPaddingCnt,         \
                                           bool dataSelectBit, uint8_t l2CacheCtl, uint64_t srcStride,                 \
                                           uint32_t dstStride)                                                         \
    {                                                                                                                  \
        CopyGm2L1AlignV2Capture capture;                                                                               \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.blockCount = blockCount;                                                                               \
        capture.blockLen = blockLen;                                                                                   \
        capture.leftPaddingCnt = leftPaddingCnt;                                                                       \
        capture.rightPaddingCnt = rightPaddingCnt;                                                                     \
        capture.dataSelectBit = dataSelectBit;                                                                         \
        capture.l2CacheCtl = l2CacheCtl;                                                                               \
        capture.srcStride = srcStride;                                                                                 \
        capture.dstStride = dstStride;                                                                                 \
        gGm2L1AlignV2Captures.push_back(capture);                                                                      \
        SimulateAlignV2DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt, dataSelectBit,        \
                                srcStride, dstStride);                                                                 \
    }                                                                                                                  \
    void CaptureCopyGmToCbufMultiND2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)            \
    {                                                                                                                  \
        CopyGm2L1ND2NzCapture capture;                                                                                 \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.loop1SrcStride = loop1_src_stride;                                                                     \
        capture.nValue = n_value;                                                                                      \
        capture.dValue = d_value;                                                                                      \
        capture.loop4SrcStride = loop4_src_stride;                                                                     \
        capture.enableSmallC0 = smallc0_en;                                                                            \
        gGm2L1ND2NzCaptures.push_back(capture);                                                                        \
        SimulateND2nzDataCopy(dst, src, loop1_src_stride, n_value, d_value, loop4_src_stride, smallc0_en);             \
    }                                                                                                                  \
    void CaptureCopyGmToCbufMultiDN2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)            \
    {                                                                                                                  \
        CopyGm2L1DN2NzCapture capture;                                                                                 \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.loop1SrcStride = loop1_src_stride;                                                                     \
        capture.nValue = n_value;                                                                                      \
        capture.dValue = d_value;                                                                                      \
        capture.loop4SrcStride = loop4_src_stride;                                                                     \
        capture.enableSmallC0 = smallc0_en;                                                                            \
        gGm2L1DN2NzCaptures.push_back(capture);                                                                        \
        SimulateDN2nzDataCopy(dst, src, loop1_src_stride, n_value, d_value, loop4_src_stride, smallc0_en);             \
    }

CAPTURE_GM_TO_L1_IMPL(uint8_t);
CAPTURE_GM_TO_L1_IMPL(half);
CAPTURE_GM_TO_L1_IMPL(uint16_t);
CAPTURE_GM_TO_L1_IMPL(float);
CAPTURE_GM_TO_L1_IMPL(uint32_t);
