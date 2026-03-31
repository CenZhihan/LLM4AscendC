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

enum TestMode {
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TEST_SELECT : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

template <typename T, TestMode TestMode, bool withMask>
void MainVecSelectLevel2Demo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ selMaskGm,
    __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, SELMODE selMode, uint32_t dataSize,
    uint32_t selMaskSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<uint8_t> selMaskGlobal;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(src1Gm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(selMaskGm), selMaskSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf);
    input0Local.InitBuffer(0, dataSize);

    LocalTensor<T> input1Local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input1Local.SetAddr(tbuf1);
    input1Local.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>), dataSize);

    LocalTensor<uint8_t> selMaskLocal;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    selMaskLocal.SetAddr(tbuf2);
    selMaskLocal.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>) + input1Local.GetSize() * sizeof(PrimT<T>), selMaskSize);

    LocalTensor<T> outputLocal;
    TBuffAddr tbuf3;
    tbuf3.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    outputLocal.SetAddr(tbuf3);
    outputLocal.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>) + input1Local.GetSize() * sizeof(PrimT<T>) +
        selMaskLocal.GetSize() * sizeof(uint8_t),
        dataSize);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);
    DataCopy(selMaskLocal, selMaskGlobal, selMaskSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (withMask) {
        if constexpr (TestMode == LEVEL2) {
            Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, dataSize);
        } else if constexpr (TestMode == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {140, 0};
            if (sizeof(PrimT<T>) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(PrimT<T>) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
                Select(outputLocal, selMaskLocal, input0Local, static_cast<PrimT<T>>(0), selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, static_cast<PrimT<T>>(0), input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local, input0Local[0], selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local[0], input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(outputLocal, selMaskLocal, input0Local, static_cast<PrimT<T>>(0), selMode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, static_cast<PrimT<T>>(0), input0Local, selMode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local, input0Local[0], selMode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local[0], input0Local, selMode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            } else {
                Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            }
        } else if constexpr (TestMode == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(PrimT<T>) == 2) {
                mask = 128;
            } else if (sizeof(PrimT<T>) == 4) {
                mask = 64;
            }
            if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
                Select(outputLocal, selMaskLocal, input0Local, static_cast<PrimT<T>>(0), selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, static_cast<PrimT<T>>(0), input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local, input0Local[0], selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local[0], input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(outputLocal, selMaskLocal, input0Local, static_cast<PrimT<T>>(0), selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, static_cast<PrimT<T>>(0), input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local, input0Local[0], selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(outputLocal, selMaskLocal, input0Local[0], input0Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            } else {
                Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(outputLocal, selMaskLocal, input0Local, input1Local, selMode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            }
        }
    } else {
        uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
        LocalTensor<uint32_t> tmpLocal;
        PopStackBuffer<uint32_t, TPosition::LCM>(tmpLocal);
        AscendC::SetVectorMask<PrimT<T>, AscendC::MaskMode::COUNTER>(static_cast<uint64_t>(122));
        AscendC::SetMaskCount();
        if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            uint32_t selAddr = static_cast<uint32_t>(
                reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t *>(selMaskLocal.GetPhyAddr())));
            Duplicate(tmpLocal, selAddr, 32);
            SetCmpMask(tmpLocal);
            Select<T, SELMODE::VSEL_TENSOR_TENSOR_MODE>(outputLocal, input0Local, input1Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            SetCmpMask(input1Local);
            Select(outputLocal, selMaskLocal, input0Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SetCmpMask(selMaskLocal);
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(outputLocal, input0Local, input1Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::SetVectorMask<PrimT<T>, AscendC::MaskMode::NORMAL>(static_cast<uint64_t>(0x00000000), static_cast<uint64_t>(0x0000ffff));
        if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            auto selAddr = reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t *>(selMaskLocal.GetPhyAddr()));
            auto tmpLocalPtr = (int64_t*)tmpLocal.GetPhyAddr();
            *tmpLocalPtr = selAddr;
            SetCmpMask(tmpLocal);
            Select<T, SELMODE::VSEL_TENSOR_TENSOR_MODE>(outputLocal, input0Local, input1Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (selMode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            SetCmpMask(input1Local);
            Select(outputLocal, selMaskLocal, input0Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            SetCmpMask(selMaskLocal);
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(outputLocal, input0Local, input1Local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        }
        
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_SELECT_LEVEL2_TESTCASE(dataType, selMode, testMode, withMask)                                              \
    TEST_F(TEST_SELECT, Select##dataType##selMode##testMode##withMask##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                            \
        uint8_t input0Gm[dataSize * sizeof(PrimT<dataType>)];                                                       \
        uint8_t input1Gm[dataSize * sizeof(PrimT<dataType>)];                                                       \
        uint8_t selMaskGm[dataSize];                                                                         \
        uint8_t outputGm[dataSize * sizeof(PrimT<dataType>)];                                                       \
                                                                                                                \
        MainVecSelectLevel2Demo<dataType, testMode, withMask>(outputGm, selMaskGm, input0Gm, input1Gm, SELMODE::selMode,\
            dataSize, selMaskSize);                                                               \
                                                                                                                \
    }

VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int8_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint8_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int8_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint8_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);


// TensorTrait Cases
using TTU16 = TensorTrait<uint16_t>;
using TTU32 = TensorTrait<uint32_t>;
using TTU64 = TensorTrait<uint64_t>;
using TTI16 = TensorTrait<int16_t>;
using TTI32 = TensorTrait<int32_t>;
using TTI64 = TensorTrait<int64_t>;
using TTHalf = TensorTrait<half>;
using TTFloat = TensorTrait<float>;
using TTBF16 = TensorTrait<bfloat16_t>;

VEC_SELECT_LEVEL2_TESTCASE(TTU64, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI64, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTFloat, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI32, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU32, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTHalf, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI16, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTBF16, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU16, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(TTFloat, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI32, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU32, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTHalf, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI16, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTBF16, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU16, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTFloat, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI32, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU32, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTHalf, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTI16, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTBF16, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(TTU16, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(TTFloat, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTI32, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTU32, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTHalf, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTI16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTBF16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTU16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTFloat, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTI32, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTU32, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTHalf, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTI16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTBF16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(TTU16, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);