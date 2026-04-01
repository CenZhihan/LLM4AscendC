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
#include "kernel_tpipe_impl.h"
#include <vector>

namespace AscendC {
class TQueWithBufIDTest : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(1);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }

    template <typename T>
    void GetData(LocalTensor<T>& tensor, uint32_t& addr, uint32_t& len, uint8_t& bufId) {
        TBufHandle handle = tensor.GetBufferHandle();
        TBufType* bufType = reinterpret_cast<TBufType*>(handle);
        addr = bufType->address;
        len = bufType->dataLen;
        bufId = bufType->bufId;
    }
};

TEST_F(TQueWithBufIDTest, testAicQueInitBuffer)
{
    TQue<TPosition::A1, 1> que;
    TSCM<TPosition::VECIN, 1> tscm;
    Hardware hardPos = GetPhyType(TPosition::VECIN);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[(uint8_t)hardPos].maxAddr, 0);
    pipe.InitBuffer(tscm, 1, len);
    pipe.InitBuffer(que, 1, len);
    EXPECT_EQ(que.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
    EXPECT_EQ(que.bufStart->address, 0);
    EXPECT_EQ(static_cast<uint8_t>(que.bufStart->enQueEvtID), INVALID_TBUFID);
    EXPECT_EQ(static_cast<uint8_t>(que.bufStart->freeBufEvtID), INVALID_TBUFID);
    EXPECT_EQ(que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(que.bufStart->dataLen, len);
    EXPECT_EQ(que.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 2);
}

TEST_F(TQueWithBufIDTest, testAicQueInitBufferWithBufferID2)
{
    static constexpr TQueConfig config = {
        .nd2nz = false,
        .nz2nd = false,
        .scmBlockGroup = false,
        .bufferLen = 0,
        .bufferNumber = 1,
        .consumerSize = 0,
        .consumer = {},
        .enableStaticEvtId = true
    };
    TQue<TPosition::A1, 1> que;
    TQue<TPosition::B1, 1, &config> que1;
    Hardware hardPos = GetPhyType(TPosition::A1);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    pipe.InitBuffer(que, 1, len);
    pipe.InitBuffer(que1, 1, len);
    EXPECT_EQ(que.bufStart, &(pipe.g_tpipeImpl.buf_[0]));
    EXPECT_EQ(que.bufStart->address, 0);
    EXPECT_EQ(static_cast<uint8_t>(que.bufStart->enQueEvtID), INVALID_TBUFID);
    EXPECT_EQ(static_cast<uint8_t>(que.bufStart->freeBufEvtID), INVALID_TBUFID);
    EXPECT_EQ(que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(que.bufStart->dataLen, len);
    EXPECT_EQ(que.bufStart->usertag, -1);
    EXPECT_EQ(que1.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
    EXPECT_EQ(que1.bufStart->address, len);
    EXPECT_EQ(que1.bufStart->state, TBufState::FREE);
    EXPECT_EQ(que1.bufStart->dataLen, len);
    EXPECT_EQ(que1.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 2);
}

TEST_F(TQueWithBufIDTest, testAicQueInitBufferWithDifferentQue)
{
    static constexpr TQueConfig config = {
        .nd2nz = false,
        .nz2nd = false,
        .scmBlockGroup = false,
        .bufferLen = 0,
        .bufferNumber = 2,
        .consumerSize = 0,
        .consumer = {},
        .enableStaticEvtId = true
    };
    TQue<TPosition::A1, 2, &config> a1Que;
    TQue<TPosition::B1, 2, &config> b1Que;
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    pipe.InitBuffer(a1Que, 2, len);
    pipe.InitBuffer(b1Que, 2, len);
    EXPECT_EQ(a1Que.bufStart, &(pipe.g_tpipeImpl.buf_[0]));
    EXPECT_EQ(a1Que.bufStart->address, 0);
    EXPECT_EQ(a1Que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(a1Que.bufStart->dataLen, len);
    EXPECT_EQ(a1Que.bufStart->usertag, -1);
    EXPECT_EQ(b1Que.bufStart, &(pipe.g_tpipeImpl.buf_[2]));
    EXPECT_EQ(b1Que.bufStart->address, 2 * len);
    EXPECT_EQ(b1Que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(b1Que.bufStart->dataLen, len);
    EXPECT_EQ(b1Que.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 4);
}

TEST_F(TQueWithBufIDTest, testAicTscmQueInitWithBufId)
{
    static constexpr TQueConfig config = {
        .nd2nz = false,
        .nz2nd = false,
        .scmBlockGroup = true,
        .bufferLen = 0,
        .bufferNumber = 1,
        .consumerSize = 0,
        .consumer = {},
        .enableStaticEvtId = true
    };
    TSCM<TPosition::GM, 1> tscmGm;
    TSCM<TPosition::VECIN, 1> tscmVecin;
    TSCM<TPosition::GM, 1, &config> tscmGmStatic;
    uint32_t len = 1024;
    TPipe pipe;
    pipe.InitBuffer(tscmGm, 1, len);
    pipe.InitBuffer(tscmVecin, 1, len);
    pipe.InitBuffer(tscmGmStatic, 1, len);
    EXPECT_EQ(tscmGm.bufStart, &(pipe.g_tpipeImpl.buf_[0]));
    EXPECT_EQ(tscmGm.bufStart->address, TOTAL_L1_SIZE - 2 * len);
    EXPECT_EQ(tscmGm.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmGm.bufStart->dataLen, len);
    EXPECT_EQ(tscmGm.bufStart->usertag, -1);
    EXPECT_EQ(tscmVecin.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
    EXPECT_EQ(tscmVecin.bufStart->address, TOTAL_L1_SIZE - 4 * len);
    EXPECT_EQ(tscmVecin.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmVecin.bufStart->dataLen, len);
    EXPECT_EQ(tscmVecin.bufStart->usertag, -1);
    EXPECT_EQ(tscmGmStatic.bufStart, &(pipe.g_tpipeImpl.buf_[2]));
    EXPECT_EQ(tscmGmStatic.bufStart->address, TOTAL_L1_SIZE - 5 * len);
    EXPECT_EQ(tscmGmStatic.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmGmStatic.bufStart->dataLen, len);
    EXPECT_EQ(tscmGmStatic.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 3);
}
} // namespace AscendC