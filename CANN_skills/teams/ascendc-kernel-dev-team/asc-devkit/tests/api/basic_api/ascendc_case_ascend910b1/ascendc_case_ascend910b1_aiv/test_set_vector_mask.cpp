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
// #include "api_check/kernel_cpu_check.h"
#include "test_utils.h"

using namespace std;
using namespace AscendC;

class TestSetVectorMaskCheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {AscendC::SetGCoreType(2);}
    virtual void TearDown() {AscendC::SetGCoreType(0);}
};

void SetUpMask(bool isNorm, uint64_t maskLen, uint64_t maskLow, uint64_t maskHigh)
{
    if (isNorm) {
        SetMaskNorm();
        if (maskLen == 1) {
            SetVectorMask<uint16_t, MaskMode::NORMAL>(maskLow);
        } else {
            SetVectorMask<uint16_t, MaskMode::NORMAL>(maskHigh, maskLow);
        }
    } else {
        SetMaskCount();
        if (maskLen == 1) {
            SetVectorMask<uint16_t, MaskMode::COUNTER>(maskLow);
        } else {
            SetVectorMask<uint16_t, MaskMode::COUNTER>(maskHigh, maskLow);
        }
    }
}

struct TestSetVectorMaskApiCheckParams {
    uint32_t dataSize;
    TPosition pos;
    uint16_t dstBlkStride;
    uint16_t srcBlkStride;
    uint16_t dstRptStride;
    uint16_t srcRptStride;
    uint8_t repeat;
    uint64_t maskHigh;
    uint64_t maskLow;
    uint64_t maskLen;
    bool isNorm;
    bool isSetMask;
    bool expect;
};

class TestSetVectorMaskApiCheckSuite : public testing::Test,
    public testing::WithParamInterface<TestSetVectorMaskApiCheckParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown() {AscendC::SetGCoreType(0); }
};

class TestSetVectorMaskNoApiCheckSuite : public testing::Test,
    public testing::WithParamInterface<TestSetVectorMaskApiCheckParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown() {AscendC::SetGCoreType(0); }
};

class TestSetVectorMaskIsSetMaskCheckSuite : public testing::Test,
    public testing::WithParamInterface<TestSetVectorMaskApiCheckParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown() {AscendC::SetGCoreType(0); }
};


class TestSetVectorMaskTensorOverflowSuite : public testing::Test,
    public testing::WithParamInterface<TestSetVectorMaskApiCheckParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown() {AscendC::SetGCoreType(0); }
};


INSTANTIATE_TEST_CASE_P(TEST_SET_VECTOR_MASK_API_CHECK, TestSetVectorMaskApiCheckSuite,
    ::testing::Values(
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   0, 0, 2,   true, true, true},
    TestSetVectorMaskApiCheckParams{ 256, TPosition::VECCALC,  1, 1, 8, 8, 1,   0xffffffffffffffff,
        0xffffffffffffffff, 2, true, true, true}
));

INSTANTIATE_TEST_CASE_P(TEST_SET_VECTOR_MASK_NO_API_CHECK, TestSetVectorMaskNoApiCheckSuite,
    ::testing::Values(
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   0, 0, 2, true, true, true}
));

INSTANTIATE_TEST_CASE_P(TEST_SET_VECTOR_MASK_ISSETMASK_CHECK, TestSetVectorMaskIsSetMaskCheckSuite,
    ::testing::Values(
    // norm模式  (isSetMask = T / F)
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 37, 2, true, false, true},
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 37, 2, true, true, true},
    // counter模式 (isSetMask = T / F)
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 37, 2, false, false, true},
    TestSetVectorMaskApiCheckParams { 256, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 37, 2, false, true, true}
));

INSTANTIATE_TEST_CASE_P(TEST_SET_VECTOR_MASK_TENSOR_OVERFLOW_CHECK, TestSetVectorMaskTensorOverflowSuite,
    ::testing::Values(
    // counter模式: 设置mask为 3 * 128, 空间只分配368个元素     纯主块场景
    TestSetVectorMaskApiCheckParams { 368, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 384, 1, false, false, false},
    TestSetVectorMaskApiCheckParams { 384, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 384, 1, false, false, true},
    // counter模式: 设置mask为 64, 空间只分配32个元素     纯尾块场景
    TestSetVectorMaskApiCheckParams { 32, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 64, 1, false, false, false},
    TestSetVectorMaskApiCheckParams { 64, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 64, 1, false, false, true},
    // counter模式: 设置mask为 3 * 128 + 37, 空间只分配416个元素     主块 + 尾块场景
    TestSetVectorMaskApiCheckParams { 416, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 421, 1, false, false, false},
    TestSetVectorMaskApiCheckParams { 421, TPosition::VECCALC, 1, 1, 8, 8, 1,   17, 421, 1, false, false, true},
    // counter模式：设置mask为 3 * 128 + 37, 空间分配大小：     主块 + 尾块场景   (仅影响尾块的时候) 37 = 2 * 16 + 5
    // 特殊用例: repStride较小，blkStride较大，最后一次主块repeat时范围大于尾块时的范围
    // main 512 + (3*7 + 1) * 32 = 1216    tail: 768 + 6 * 32 + 10 = 970
    TestSetVectorMaskApiCheckParams { 485, TPosition::VECCALC, 1, 3, 8, 8, 1,   17, 421, 1, false, false, false},
    TestSetVectorMaskApiCheckParams { 608, TPosition::VECCALC, 1, 3, 8, 8, 1,   17, 421, 1, false, false, true},
    // tailRepeatStart + tailBlkSize > mainBlkSize的用例  主块 + 尾块 3 * 128 + 101 (6 * 16 + 5)
    // main: 512 + 512  tail: 768 + 384 + 10
    TestSetVectorMaskApiCheckParams { 512, TPosition::VECCALC, 1, 2, 8, 8, 1,   17, 485, 1, false, false, false},
    TestSetVectorMaskApiCheckParams { 581, TPosition::VECCALC, 1, 2, 8, 8, 1,   17, 485, 1, false, false, true}
));

// when mask both 0, call api will not cause error in 910B1
TEST_P(TestSetVectorMaskApiCheckSuite, SetVectorMaskApiCheckLowLevel)
{
    TPipe tpipe;
    auto param = GetParam();
    uint32_t dataSize = param.dataSize;
    uint64_t maskHigh = param.maskHigh;
    uint64_t maskLow = param.maskLow;
    LocalTensor<uint16_t> input0;
    LocalTensor<uint16_t> input1;
    LocalTensor<uint16_t> output;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input0 = tbuf.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input1 = tbuf1.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    output = tbuf2.Get<uint16_t>();

    UnaryRepeatParams repeatParams { param.dstBlkStride, param.srcBlkStride, param.dstRptStride, param.srcRptStride };
    uint8_t repeatTimes = param.repeat;
    uint64_t mask = 128;
    uint64_t maskFull = 0xffffffffffffffff;
    check::VecBinaryScalarApiParams chkParams { reinterpret_cast<uint64_t>(output.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input0.GetPhyAddr()),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint64_t>(output.GetLength()),
        static_cast<uint64_t>(input0.GetLength()),
        static_cast<uint8_t>(output.GetPosition()),
        static_cast<uint8_t>(input0.GetPosition()) };
    // check::TikcppVecBinaryScalarCheck chkIns { "test_intri", chkParams };
    // MaskSetter::Instance().SetMask(true);
    // bool flag = chkIns.CheckAllLowLevel({ maskHigh, maskLow });
    // EXPECT_EQ(flag, param.expect);
}

// when mask both 0, do not call api will not cause error
TEST_P(TestSetVectorMaskNoApiCheckSuite, SetVectorMaskNoApiCheckLowLevel)
{
    TPipe tpipe;
    auto param = GetParam();
    uint32_t dataSize = param.dataSize;
    uint64_t maskHigh = param.maskHigh;
    uint64_t maskLow = param.maskLow;
    SetMaskNorm();
    SetVectorMask<uint16_t, MaskMode::NORMAL>(maskHigh, maskLow);
}

// In counter / norm mode, if setmask = false, update mask with mask value in registers.
TEST_P(TestSetVectorMaskIsSetMaskCheckSuite, UpdateMaskArrayAndCheck)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t repeatTimes = param.repeat;
    uint32_t dataSize = param.dataSize;
    uint64_t maskHigh = param.maskHigh;
    uint64_t maskLow = param.maskLow;
    uint64_t maskLen = param.maskLen;
    bool isNorm = param.isNorm;
    bool isSetMask = param.isSetMask;
    LocalTensor<uint16_t> input0;
    LocalTensor<uint16_t> input1;
    LocalTensor<uint16_t> output;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input0 = tbuf.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input1 = tbuf1.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    output = tbuf2.Get<uint16_t>();

    UnaryRepeatParams repeatParams {param.dstBlkStride, param.srcBlkStride, param.dstRptStride, param.srcRptStride};
    SetUpMask(isNorm, maskLen, maskLow, maskHigh);

    // if setMask = true, return original userInputMaskArray
    uint64_t userMask = 117;
    uint64_t userMaskLow = 109;
    uint64_t userMaskHigh = 303;
    std::vector<uint64_t> userInputMaskArray = {userMask};
    if (maskLen == 2) {
        userInputMaskArray = {userMaskHigh, userMaskLow};
    }
    std::vector<uint64_t> originalUserMaskArray = userInputMaskArray;
    MaskSetter::Instance().SetMask(isSetMask);

    check::VecBinaryScalarApiParams chkParams { reinterpret_cast<uint64_t>(output.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input0.GetPhyAddr()),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint64_t>(output.GetLength()),
        static_cast<uint64_t>(input0.GetLength()),
        static_cast<uint8_t>(output.GetPosition()),
        static_cast<uint8_t>(input0.GetPosition()) };
    // check::TikcppVecBinaryScalarCheck chkIns { "test_intri", chkParams };

    // bool res = chkIns.UpdateMaskArrayAndCheck(userInputMaskArray, sizeof(uint16_t));
    // // 如果isSetMask是False，那么UpdateMaskArrayAndCheck会将maskArray刷新成寄存器中的maskHigh + maskLow
    // if (!isSetMask) {
    //     EXPECT_EQ(userInputMaskArray[0], maskHigh);
    //     EXPECT_EQ(userInputMaskArray[1], maskLow);
    // } else { // maskArray不变，保持和用户传入的一致
    //     EXPECT_EQ(originalUserMaskArray, userInputMaskArray);
    // }
}

// In counter, check with tensor overflow scences
TEST_P(TestSetVectorMaskTensorOverflowSuite, CheckTensorOverflow)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t repeatTimes = param.repeat;
    uint32_t dataSize = param.dataSize;
    uint64_t maskHigh = param.maskHigh;
    uint64_t maskLow = param.maskLow;
    uint64_t maskLen = param.maskLen;
    bool isNorm = param.isNorm;
    bool isSetMask = param.isSetMask;
    LocalTensor<uint16_t> input0;
    LocalTensor<uint16_t> input1;
    LocalTensor<uint16_t> output;

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input0 = tbuf.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    input1 = tbuf1.Get<uint16_t>();
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
    output = tbuf2.Get<uint16_t>();

    UnaryRepeatParams repeatParams {param.dstBlkStride, param.srcBlkStride, param.dstRptStride, param.srcRptStride};
    SetUpMask(isNorm, maskLen, maskLow, maskHigh);

    // if setMask = true, return original userInputMaskArray
    uint64_t userMask = 117;
    uint64_t userMaskLow = 109;    //      0110 1101
    uint64_t userMaskHigh = 303;   // 0001 0010 1111
    std::vector<uint64_t> userInputMaskArray = {userMask};
    if (maskLen == 2) {
        userInputMaskArray = {userMaskHigh, userMaskLow};
    }
    std::vector<uint64_t> originalUserMaskArray = userInputMaskArray;
    MaskSetter::Instance().SetMask(isSetMask);

    check::VecBinaryScalarApiParams chkParams { reinterpret_cast<uint64_t>(output.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input0.GetPhyAddr()),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint64_t>(output.GetLength()),
        static_cast<uint64_t>(input0.GetLength()),
        static_cast<uint8_t>(output.GetPosition()),
        static_cast<uint8_t>(input0.GetPosition()) };
    // check::TikcppVecBinaryScalarCheck chkIns { "test_intri", chkParams };

    // bool res = chkIns.UpdateMaskArrayAndCheck(userInputMaskArray, sizeof(uint16_t));
    // check::TensorOverflowParams params = {input0.GetLength(), sizeof(uint16_t), static_cast<uint64_t>(repeatTimes),
    //     static_cast<uint64_t>(repeatParams.srcBlkStride), static_cast<uint64_t>(repeatParams.srcRepStride), false};
    // res = chkIns.CheckTensorOverflowLow(userInputMaskArray, params, "Error");
    // EXPECT_EQ(res, param.expect);
}