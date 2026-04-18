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
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, 0);
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

TEST_F(TQueWithBufIDTest, testAicQueInitBufferWithBufferId)
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
    TQue<TPosition::A1, 1, &config> que;
    TSCM<TPosition::VECIN, 1> tscm;
    Hardware hardPos = GetPhyType(TPosition::A1);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, 0);
    pipe.InitBuffer(tscm, 1, len);
    pipe.InitBuffer(que, 1, len);
    EXPECT_EQ(que.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
    EXPECT_EQ(que.bufStart->address, 0);
    EXPECT_EQ(que.bufStart->bufId, 0);
    EXPECT_EQ(que.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(que.bufStart->dataLen, len);
    EXPECT_EQ(que.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 2);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, len);

    auto tensor = que.AllocTensor<float>();
    EXPECT_EQ(que.bufStart->bufId, 0);
    que.EnQue(tensor);
    EXPECT_EQ(que.bufStart->bufId, 0);
    que.DeQue<float>();
    EXPECT_EQ(que.bufStart->bufId, 0);
    que.FreeTensor(tensor);
    EXPECT_EQ(que.bufStart->bufId, 0);
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
    EXPECT_EQ(que1.bufStart->bufId, 0);
    EXPECT_EQ(que1.bufStart->bufIdAlt, INVALID_TBUFID);
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
    EXPECT_EQ(a1Que.bufStart->bufId, 0);
    EXPECT_EQ(a1Que.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(a1Que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(a1Que.bufStart->dataLen, len);
    EXPECT_EQ(a1Que.bufStart->usertag, -1);
    EXPECT_EQ(b1Que.bufStart, &(pipe.g_tpipeImpl.buf_[2]));
    EXPECT_EQ(b1Que.bufStart->address, 2 * len);
    EXPECT_EQ(b1Que.bufStart->bufId, 2);
    EXPECT_EQ(b1Que.bufStart->bufIdAlt, INVALID_TBUFID);
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
    EXPECT_EQ(tscmGm.bufStart->bufId, 0);
    EXPECT_EQ(tscmGm.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(tscmGm.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmGm.bufStart->dataLen, len);
    EXPECT_EQ(tscmGm.bufStart->usertag, -1);
    EXPECT_EQ(tscmVecin.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
    EXPECT_EQ(tscmVecin.bufStart->address, TOTAL_L1_SIZE - 4 * len);
    EXPECT_EQ(tscmVecin.bufStart->bufId, 11);
    EXPECT_EQ(tscmVecin.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(tscmVecin.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmVecin.bufStart->dataLen, len);
    EXPECT_EQ(tscmVecin.bufStart->usertag, -1);
    EXPECT_EQ(tscmGmStatic.bufStart, &(pipe.g_tpipeImpl.buf_[2]));
    EXPECT_EQ(tscmGmStatic.bufStart->address, TOTAL_L1_SIZE - 5 * len);
    EXPECT_EQ(tscmGmStatic.bufStart->bufId, 1);
    EXPECT_EQ(tscmGmStatic.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(tscmGmStatic.bufStart->state, TBufState::FREE);
    EXPECT_EQ(tscmGmStatic.bufStart->dataLen, len);
    EXPECT_EQ(tscmGmStatic.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 3);
}

TEST_F(TQueWithBufIDTest, testCustomizeInitBuffer)
{
    TPipe pipe;
    TQue<TPosition::B1, 1> que1;
    TQue<TPosition::A1, 1> que2;
    TQue<TPosition::A2, 1> que3;
    auto addr0 = Std::make_tuple(0, 1024);
    auto addr1 = Std::make_tuple(1024, 1024);
    auto addr2 = Std::make_tuple(2048, 1024);
    auto addr3 = Std::make_tuple(3072, 1024);
    auto addr4 = Std::make_tuple(4096, 1024);
    auto addr5 = Std::make_tuple(5120, 1024);

    pipe.InitBuffer(que1, addr0, addr1, addr2);
    pipe.InitBuffer(que2, addr3);
    pipe.InitBuffer(que3, addr4, addr5);

    TBufHandle handle;
    TBufType* buftype;
    LocalTensor<uint8_t> tensor;

    tensor = que1.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 0);

    tensor = que1.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 1024);

    tensor = que1.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 2048);

    tensor = que2.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 3072);

    tensor = que3.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 4096);

    tensor = que3.AllocTensor<uint8_t>();
    buftype = (TBufType*)tensor.GetBufferHandle();
    EXPECT_EQ(tensor.GetSize(), 1024);
    EXPECT_EQ(buftype->address, 5120);
}

TEST_F(TQueWithBufIDTest, testGlobalManageQue)
{
    constexpr uint32_t blockLen = 64 * 1024;
    static constexpr TQueConfig config = {
        .nd2nz = false,
        .nz2nd = false,
        .scmBlockGroup = false,
        .bufferLen = blockLen,
        .bufferNumber = 4
    };
    TQue<TPosition::CO1, 1, &config> que;
    TPipe pipe;
    pipe.InitBuffer(que, 2, blockLen);

    uint32_t addr = 0;
    uint32_t len = 0;
    uint8_t bufId = 0;

    auto t2 = que.AllocTensor<float>(2);
    GetData(t2, addr, len, bufId);
    EXPECT_EQ(addr, 0);
    EXPECT_EQ(len, 2 * blockLen);
    EXPECT_EQ(bufId, 20);
    auto t = que.AllocTensor<float>(1);
    GetData(t, addr, len, bufId);
    EXPECT_EQ(addr, 2 * blockLen);
    EXPECT_EQ(len, blockLen);
    EXPECT_EQ(bufId, 22);

    que.EnQue(t);
    que.DeQue<float>();
    que.EnQue(t2);
    que.DeQue<float>();
    que.FreeTensor(t);
    que.FreeTensor(t2);

    auto t3 = que.AllocTensor<float>(2);
    GetData(t3, addr, len, bufId);
    EXPECT_EQ(addr, 0);
    EXPECT_EQ(len, 2 * blockLen);
    EXPECT_EQ(bufId, 20);
    que.EnQue(t3);
    que.DeQue<float>();
    que.FreeTensor(t3);
}

TEST_F(TQueWithBufIDTest, testTBufInitBuffer)
{
    constexpr uint32_t blockLen = 1024;
    static constexpr TQueConfig config = {
        .nd2nz = false,
        .nz2nd = false,
        .scmBlockGroup = true,
        .bufferLen = 0,
        .bufferNumber = 2,
        .consumerSize = 0,
        .consumer = {},
        .enableStaticEvtId = true
    };
    TQue<TPosition::A1, 1, &config> que;
    TPipe pipe;
    TBufPool<TPosition::A1> bufPool;
    pipe.InitBufPool(bufPool, 4 * blockLen);
    bufPool.InitBuffer(que, 2, blockLen);

    uint32_t addr = 0;
    uint32_t len = 0;
    uint8_t bufId = 0;

    for (uint32_t i = 0; i < 2; i++) {
        auto t = que.AllocTensor<float>(1);
        GetData(t, addr, len, bufId);
        EXPECT_EQ(addr, blockLen * (i % 2));
        EXPECT_EQ(len, blockLen);
        EXPECT_EQ(bufId, i);
        que.EnQue(t);
        que.DeQue<float>();
        que.FreeTensor(t);
    }
}

TEST_F(TQueWithBufIDTest, testGetTQueHeadAddr)
{
    constexpr uint32_t blockLen = 1024;
    TQue<TPosition::A1, 1> que1;
    TQue<TPosition::A1, 1> que2;
    TPipe pipe;
    pipe.InitBuffer(que1, 2, blockLen);
    pipe.InitBuffer(que2, 2, blockLen);

    EXPECT_EQ(GetTQueHeadAddr(que1), 0);
    EXPECT_EQ(GetTQueHeadAddr(que2), 2 * blockLen);
}

TEST_F(TQueWithBufIDTest, testTPipeSharedMode)
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
    TQue<TPosition::A1, 1, &config> que1;
    TQue<TPosition::A1, 1, &config> que2;
    constexpr uint32_t len = 1024;
    constexpr uint32_t sharedLen = 10240;
    uint32_t shareLens[3] = {sharedLen, sharedLen, sharedLen};
    TPipe pipe;
    uint32_t addr = 0;
    uint32_t actualLen = 0;
    uint8_t bufId = 0;
    InitShareBufStart(&pipe, 0, shareLens, 3, 0);
    pipe.InitBuffer(que1, 1, len);
    auto t = que1.AllocTensor<float>();
    GetData(t, addr, actualLen, bufId);
    EXPECT_EQ(addr, 0);
    EXPECT_EQ(actualLen, len);
    EXPECT_EQ(bufId, 0);
    que1.EnQue(t);
    que1.DeQue<float>();
    que1.FreeTensor(t);
    InitShareBufEnd(&pipe);

    InitShareBufStart(&pipe, 0, shareLens, 3, 0);
    pipe.InitBuffer(que2, 1, len);
    t = que2.AllocTensor<float>();
    GetData(t, addr, actualLen, bufId);
    EXPECT_EQ(addr, 0);
    EXPECT_EQ(actualLen, len);
    EXPECT_EQ(bufId, 0);
    que2.EnQue(t);
    que2.DeQue<float>();
    que2.FreeTensor(t);
    InitShareBufEnd(&pipe);
}
} // namespace AscendC