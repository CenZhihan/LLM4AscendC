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

class TestReducev2Check : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

struct TestReducev2ApiCheckParams {
    bool reduceMode;
    uint32_t dataSize;
    TPosition pos;
    uint16_t srcBlkStride;
    uint16_t srcRptStride;
    uint8_t repeat;
    uint32_t mask;
    bool expect;
};

struct TestReducev2ScalarApiCheckParams {
    bool reduceMode;
    uint32_t dataSize;
    TPosition pos;
    uint16_t srcBlkStride;
    uint16_t srcRptStride;
    uint8_t repeat;
    uint32_t mask;
    bool expect;
};

class TestReducev2ApiCheckSuite : public testing::Test, public testing::WithParamInterface<TestReducev2ApiCheckParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

class TestReducev2ScalarApiCheckSuite : public testing::Test,
    public testing::WithParamInterface<TestReducev2ScalarApiCheckParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_REDUCEV2_API_CHECK, TestReducev2ApiCheckSuite,
    ::testing::Values(
    // norm模式
    TestReducev2ApiCheckParams {false, 512, TPosition::VECCALC,           1, 8, 1,   8,  true },
    TestReducev2ApiCheckParams {false, 512, TPosition::A1,                1, 8, 1,   8,  false },
    TestReducev2ApiCheckParams {false, 16 * 1024 + 1, TPosition::VECCALC, 1, 8, 1,   8,  true },
    TestReducev2ApiCheckParams {false, 512, TPosition::VECCALC,           1, 8, 4,   8,  true },
    // counter模式， repeatTimes较为特殊，需要生效
    TestReducev2ApiCheckParams {true, 8, TPosition::VECCALC,              1, 0, 1,   8, true },  // repeatTimes = 1
    // repeatTimes > 1 + blkStride > 1
    // 最后一次rep起始位置: (3-1) * 7 * 32 = 448   需要block数： 100 / 16 = 6个block + 4个数
    // block偏移量： 6 * 4 * 32 = 768    总共 768 + 448 + 4 * 2 = 1224
    TestReducev2ApiCheckParams {true, 612, TPosition::VECCALC,              4, 7, 3,   100, true },
    TestReducev2ApiCheckParams {true, 608, TPosition::VECCALC,              4, 7, 3,   100, false }
));

INSTANTIATE_TEST_CASE_P(TEST_REDUCEV2_SCALAR_API_CHECK, TestReducev2ScalarApiCheckSuite,
    ::testing::Values(
    // norm模式
    // norm模式时，mask不生效。所以至少256B
    TestReducev2ScalarApiCheckParams {false,  32, TPosition::VECCALC,            1, 8, 1,   8, false },
    TestReducev2ScalarApiCheckParams {false,  128, TPosition::VECCALC,           1, 8, 1,   8, true },
    TestReducev2ScalarApiCheckParams {false,  512, TPosition::A1,                1, 8, 1,   8, false },
    TestReducev2ScalarApiCheckParams {false,  16 * 1024 + 1, TPosition::VECCALC, 1, 8, 1,   8, true },
    // repeatTimes > 1
    TestReducev2ScalarApiCheckParams {false,  128, TPosition::VECCALC,           1, 8, 4,   8, false },
    TestReducev2ScalarApiCheckParams {false,  512, TPosition::VECCALC,           1, 8, 4,   8, true }
));

TEST_P(TestReducev2ApiCheckSuite, ReApiCheckTensor)
{
    TPipe tpipe;
    auto param = GetParam();
    uint32_t dataSize = param.dataSize;
    bool reduceMode = param.reduceMode;
    uint64_t mask = param.mask;
    LocalTensor<uint16_t> input0;
    LocalTensor<uint16_t> input1;
    LocalTensor<uint16_t> output;
    if (param.pos == TPosition::VECCALC) {
        TBuf<TPosition::VECCALC> tbuf;
        tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input0 = tbuf.Get<uint16_t>();

        TBuf<TPosition::VECCALC> tbuf1;
        tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input1 = tbuf1.Get<uint16_t>();

        TBuf<TPosition::VECCALC> tbuf2;
        tpipe.InitBuffer(tbuf2, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        output = tbuf2.Get<uint16_t>();
    } else {
        TBuf<TPosition::A1> tbuf;
        tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input0 = tbuf.Get<uint16_t>();

        TBuf<TPosition::A1> tbuf1;
        tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input1 = tbuf1.Get<uint16_t>();

        TBuf<TPosition::A1> tbuf2;
        tpipe.InitBuffer(tbuf2, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        output = tbuf2.Get<uint16_t>();
    }

    GatherMaskParams repeatParams { param.srcBlkStride, param.repeat, param.srcRptStride, param.srcRptStride };
    uint64_t rsvdCnt = 0;
    check::VecGatherMaskApiParams chkParams { reinterpret_cast<uint64_t>(output.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input0.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input1.GetPhyAddr()),
        static_cast<bool>(reduceMode),
        static_cast<uint16_t>(repeatParams.src0BlockStride),
        repeatParams.repeatTimes,
        static_cast<uint16_t>(repeatParams.src0RepeatStride),
        static_cast<uint16_t>(repeatParams.src1RepeatStride),
        static_cast<uint64_t>(rsvdCnt),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint64_t>(output.GetLength()),
        static_cast<uint64_t>(input0.GetLength()),
        static_cast<uint64_t>(input1.GetLength()),
        static_cast<uint8_t>(output.GetPosition()),
        static_cast<uint8_t>(input0.GetPosition()),
        static_cast<uint8_t>(input1.GetPosition()) };
    // check::TikcppGatherMaskCheck chkIns { "test_intri", chkParams };
    // bool flag = chkIns.CheckAllLowLevel({ 0, mask });
    // EXPECT_EQ(flag, param.expect);
}

TEST_P(TestReducev2ScalarApiCheckSuite, ReApiCheckScalar)
{
    TPipe tpipe;
    auto param = GetParam();
    uint32_t dataSize = param.dataSize;
    bool reduceMode = param.reduceMode;
    uint64_t mask = param.mask;
    LocalTensor<uint16_t> input0;
    LocalTensor<uint16_t> output;
    if (param.pos == TPosition::VECCALC) {
        TBuf<TPosition::VECCALC> tbuf;
        tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input0 = tbuf.Get<uint16_t>();

        TBuf<TPosition::VECCALC> tbuf1;
        tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        output = tbuf1.Get<uint16_t>();
    } else {
        TBuf<TPosition::A1> tbuf;
        tpipe.InitBuffer(tbuf, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        input0 = tbuf.Get<uint16_t>();

        TBuf<TPosition::A1> tbuf1;
        tpipe.InitBuffer(tbuf1, ALIGN_ADDR(dataSize * sizeof(uint16_t)));
        output = tbuf1.Get<uint16_t>();
    }

    GatherMaskParams repeatParams { param.srcBlkStride, param.repeat, param.srcRptStride, param.srcRptStride };
    uint64_t rsvdCnt = 0;
    uint8_t src1Pattern = 1;

    check::VecGatherMaskApiParams chkParams { reinterpret_cast<uint64_t>(output.GetPhyAddr()),
        reinterpret_cast<uint64_t>(input0.GetPhyAddr()),
        static_cast<uint8_t>(src1Pattern),
        static_cast<bool>(reduceMode),
        static_cast<uint16_t>(repeatParams.src0BlockStride),
        repeatParams.repeatTimes,
        static_cast<uint16_t>(repeatParams.src0RepeatStride),
        static_cast<uint16_t>(repeatParams.src1RepeatStride),
        static_cast<uint64_t>(rsvdCnt),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint32_t>(sizeof(uint16_t)),
        static_cast<uint64_t>(output.GetLength()),
        static_cast<uint64_t>(input0.GetLength()),
        static_cast<uint8_t>(output.GetPosition()),
        static_cast<uint8_t>(input0.GetPosition()) };
    // check::TikcppGatherMaskCheck chkIns { "test_intri", chkParams };
    // bool flag = chkIns.CheckAllLowLevel({ 0, mask });
    // EXPECT_EQ(flag, param.expect);
}