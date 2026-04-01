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
#include "mockcpp/mockcpp.hpp"
#include <vector>

namespace AscendC {
namespace {
constexpr int32_t TILE_NUM = 8;                               // split data into 8 tiles for each core
constexpr int32_t TILE_LENGTH = 2048;      // length of each tile
constexpr int32_t BUFFER_LENGTH = 1024;      // length of each tile
constexpr int32_t BLOCK_LENGTH = TILE_NUM * TILE_LENGTH;                      // length computed of each core
constexpr int32_t BUFFER_NUM = 2;                             // tensor num for each queue
constexpr int32_t QUE_DEPTH = 1;                          // tensor num for each queue
constexpr int32_t BUF_ID_SIZE = 4;                          // tensor num for each queue
}

template <typename T>
void GetData(LocalTensor<T>& tensor, uint32_t& addr, uint32_t& len, uint8_t& bufId) {
    TBufHandle handle = tensor.GetBufferHandle();
    TBufType* bufType = reinterpret_cast<TBufType*>(handle);
    addr = bufType->address;
    len = bufType->dataLen;
    bufId = bufType->bufId;
}

class TQueWithBufIDTest : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

TEST_F(TQueWithBufIDTest, testAivQueInitBufferWithBufferID)
{
    TQue<TPosition::VECIN, 1> que;
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
    EXPECT_EQ(que.bufStart->bufId, 0);
    EXPECT_EQ(que.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(que.bufStart->state, TBufState::FREE);
    EXPECT_EQ(que.bufStart->dataLen, len);
    EXPECT_EQ(que.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 2);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, len);

    uint32_t lenGet = 32;
    auto tensor = que.AllocTensor<float>();
    que.EnQue(tensor);
    que.DeQue<float>();
    que.FreeTensor(tensor);
    EXPECT_EQ(que.bufStart->bufId, 0);
    EXPECT_EQ(que.VacantInQue(), true);
    EXPECT_EQ(que.HasTensorInQue(), false);
    EXPECT_EQ(que.GetTensorCountInQue(), 0);
    EXPECT_EQ(que.HasIdleBuffer(), true);
}

TEST_F(TQueWithBufIDTest, testAivQueInitMultipleBuffers)
{
    constexpr int32_t size = 28;
    TQue<TPosition::VECIN, 1> que[size];
    Hardware hardPos = GetPhyType(TPosition::VECIN);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, 0);
    for (int32_t i = 0; i < size; i++) {
        pipe.InitBuffer(que[i], 1, len);
        EXPECT_EQ(que[i].bufStart, &(pipe.g_tpipeImpl.buf_[i]));
        EXPECT_EQ(que[i].bufStart->address, i * len);
        EXPECT_EQ(que[i].bufStart->bufId, i);
        EXPECT_EQ(que[i].bufStart->bufIdAlt, INVALID_TBUFID);
        EXPECT_EQ(que[i].bufStart->state, TBufState::FREE);
        EXPECT_EQ(que[i].bufStart->dataLen, len);
        EXPECT_EQ(que[i].bufStart->usertag, -1);
        EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, i + 1);
    }
}

TEST_F(TQueWithBufIDTest, testAivQueInitBufferWithDifferentQue)
{
    TQue<TPosition::VECIN, 1> inQue;
    TQue<TPosition::VECOUT, 1> outQue;
    Hardware hardPos = GetPhyType(TPosition::A1);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, 0);
    pipe.InitBuffer(inQue, 2, len);
    pipe.InitBuffer(outQue, 2, len);
    EXPECT_EQ(inQue.bufStart, &(pipe.g_tpipeImpl.buf_[0]));
    EXPECT_EQ(inQue.bufStart->address, 0);
    EXPECT_EQ(inQue.bufStart->bufId, 0);
    EXPECT_EQ(inQue.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(inQue.bufStart->state, TBufState::FREE);
    EXPECT_EQ(inQue.bufStart->dataLen, len);
    EXPECT_EQ(inQue.bufStart->usertag, -1);
    EXPECT_EQ(outQue.bufStart, &(pipe.g_tpipeImpl.buf_[2]));
    EXPECT_EQ(outQue.bufStart->address, 2 * len);
    EXPECT_EQ(outQue.bufStart->bufId, 2);
    EXPECT_EQ(outQue.bufStart->bufIdAlt, INVALID_TBUFID);
    EXPECT_EQ(outQue.bufStart->state, TBufState::FREE);
    EXPECT_EQ(outQue.bufStart->dataLen, len);
    EXPECT_EQ(outQue.bufStart->usertag, -1);
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 4);
}

TEST_F(TQueWithBufIDTest, testAivQueInitBufferWithNoneUBQue)
{
    TQue<TPosition::A1, 1> inQue;
    TQue<TPosition::B1, 1> outQue;
    Hardware hardPos = GetPhyType(TPosition::A1);
    uint32_t len = 1024;
    TPipe pipe;
    EXPECT_EQ(pipe.g_tpipeImpl.curBufSize_, 0);
    EXPECT_EQ(pipe.g_tpipeImpl.bufPool_[static_cast<uint8_t>(hardPos)].maxAddr, 0);
    pipe.InitBuffer(inQue, 2, len);
    pipe.InitBuffer(outQue, 2, len);
    EXPECT_EQ(Internal::g_bufId, 0xF);
}

TEST_F(TQueWithBufIDTest, testGetTQueHeadAddr)
{
    TQue<TPosition::VECIN, 1> inQue;
    TQue<TPosition::VECOUT, 1> outQue;
    bool aa = TQue<TPosition::VECIN, 1>::isTQue;
    constexpr uint32_t blockLen = 1024;
    TPipe pipe;
    pipe.InitBuffer(inQue, 2, blockLen);
    pipe.InitBuffer(outQue, 2, blockLen);

    EXPECT_EQ(GetTQueHeadAddr(inQue), 0);
    EXPECT_EQ(GetTQueHeadAddr(outQue), 2 * blockLen);
}

// TEST_F(TQueWithBufIDTest, testTscmVecinUsed)
// {
//     TSCM<TPosition::VECIN, 1> tscm1;
//     TSCM<TPosition::VECIN, 1> tscm2;
//     constexpr uint32_t len = 1024;
//     TPipe pipe;
//     pipe.InitBuffer(tscm1, 1, len);
//     pipe.InitBuffer(tscm2, 1, len);

//     uint64_t spySetBlockId = 0xff;
//     uint64_t spyWaitBlockId = 0xff;

//     MOCKER_CPP(set_intra_block, void(*)(pipe_t, uint64_t))
//         .stubs()
//         .with(any(), spy(spySetBlockId));
//     MOCKER_CPP(wait_intra_block, void(*)(pipe_t, uint64_t))
//         .stubs()
//         .with(any(), spy(spyWaitBlockId));
//     for (int32_t i = 0; i < 2; i++) {
//         auto tensor1 = tscm1.AllocTensor<float>();
//         if (i > 0) {

//         EXPECT_EQ(spyWaitBlockId, 11);
//         spyWaitBlockId = 0xff;
//         }
//         auto tensor2 = tscm2.AllocTensor<float>();
//         if (i > 0) {
//         EXPECT_EQ(spyWaitBlockId, 10);
//         spyWaitBlockId = 0xff;
//         }
//         tscm1.EnQue(tensor1);
//         tscm2.EnQue(tensor2);
//         tensor1 = tscm1.DeQue<float>();
//         EXPECT_EQ(spySetBlockId, 11);
//         spySetBlockId = 0xff;
//         tensor2 = tscm2.DeQue<float>();
//         EXPECT_EQ(spySetBlockId, 10);
//         spySetBlockId = 0xff;
//         tscm1.FreeTensor(tensor1);
//         tscm2.FreeTensor(tensor2);
//     }

//     Hardware hardPos = GetPhyType(TPosition::TSCM);
//     EXPECT_EQ(tscm1.bufStart, &(pipe.g_tpipeImpl.buf_[0]));
//     EXPECT_EQ(tscm2.bufStart, &(pipe.g_tpipeImpl.buf_[1]));
//     EXPECT_EQ(pipe.g_tpipeImpl.buf_[0].freeBufEvtID, 0);
//     EXPECT_EQ(pipe.g_tpipeImpl.buf_[1].freeBufEvtID, 0);
// }

TEST_F(TQueWithBufIDTest, testCustomizeInitBuffer)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1> que1;
    TQue<TPosition::VECOUT, 1> que2;
    TQue<TPosition::VECIN, 1> que3;
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

class TEST_TBUFPOOL : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};
 
/* **************************** Tpipe InitBufPool api ****************************** */
TEST_F(TEST_TBUFPOOL, TpipeInitBufPool)
{
    TPipe pipe;
    TBufPool<TPosition::VECOUT> tbufPool1, tbufPool2;
    TQue<TPosition::VECOUT, 2> que1, que2;
    Hardware hardPos = GetPhyType(TPosition::VECOUT);
    uint8_t num = 2;
    uint32_t len = 128;
    uint32_t poolLen = 65536;
    pipe.InitBufPool(tbufPool1, poolLen);
    pipe.InitBufPool(tbufPool2, poolLen, tbufPool1);
 
    tbufPool1.InitBuffer(que1, num, len);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.curBufSize_, 2);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.maxAddr_, num * len);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.maxLen_, poolLen);
    tbufPool2.InitBuffer(que2, num, len);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.curBufSize_, 2);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.maxAddr_, num * len);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.maxLen_, poolLen);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.startAddr_, 0);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.startAddr_, 0);
 
    tbufPool1.Reset();
    EXPECT_EQ(tbufPool1.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.maxAddr_, 0);
    tbufPool2.Reset();
    EXPECT_EQ(tbufPool2.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.maxAddr_, 0);
}
 
/* **************************** TbufPool InitBufPool api ****************************** */
TEST_F(TEST_TBUFPOOL, TbufPoolInitBufPool)
{
    TPipe pipe;
    TBufPool<TPosition::VECCALC, 6> tbufPool1;
    TBufPool<TPosition::VECCALC> tbufPool2, tbufPool3;
    TQue<TPosition::VECOUT, 2> que1;
    Hardware hardPos = GetPhyType(TPosition::VECOUT);
    uint8_t num = 2;
    uint32_t len = 128;
    uint32_t poolLen = 65536;
    pipe.InitBufPool(tbufPool1, poolLen * 2);
 
    tbufPool1.InitBuffer(que1, num, len);
    tbufPool1.InitBufPool(tbufPool2, poolLen);
    tbufPool1.InitBufPool(tbufPool3, poolLen, tbufPool2);

    EXPECT_EQ(tbufPool1.tBufPoolImpl.curBufSize_, 6); // InitBufPool doesn't generate bufHandle;
    EXPECT_EQ(tbufPool1.tBufPoolImpl.startAddr_, 0);
    EXPECT_EQ(tbufPool1.tBufPoolImpl.maxAddr_, num * len + poolLen);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.startAddr_, num * len);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.maxAddr_, num * len);
    EXPECT_EQ(tbufPool2.tBufPoolImpl.maxLen_, poolLen);
    EXPECT_EQ(tbufPool3.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(tbufPool3.tBufPoolImpl.startAddr_, num * len);
    EXPECT_EQ(tbufPool3.tBufPoolImpl.maxAddr_, num * len);
    EXPECT_EQ(tbufPool3.tBufPoolImpl.maxLen_, poolLen);
}
 
/* **************************** TbufPool InitBuffer api ****************************** */
TEST_F(TEST_TBUFPOOL, TbufPoolInitBuffer)
{
    TPipe pipe;
    TBufPool<TPosition::VECIN> tbufPool;
    TQue<TPosition::VECOUT, 2> que;
    TBuf<TPosition::VECIN> tbuf;
 
    Hardware hardPos = GetPhyType(TPosition::VECOUT);
    uint8_t num = 2;
    uint32_t poolLen = 65536;
    pipe.InitBufPool(tbufPool, poolLen);
 
    tbufPool.InitBuffer(que, num, BUFFER_LENGTH);
    tbufPool.InitBuffer(tbuf, BUFFER_LENGTH);
    EXPECT_EQ(tbufPool.tBufPoolImpl.curBufSize_, 3);
    EXPECT_EQ(tbufPool.tBufPoolImpl.startAddr_, 0);
    EXPECT_EQ(tbufPool.tBufPoolImpl.maxAddr_, 3 * BUFFER_LENGTH);
    EXPECT_EQ(tbufPool.tBufPoolImpl.maxLen_, poolLen);

    LocalTensor<float> tensor1 = que.AllocTensor<float>();
    LocalTensor<float> tensor2 = que.AllocTensor<float>();
    uint32_t actualAddr;
    uint32_t actualLen;
    uint8_t actualBufId;
    GetData(tensor1, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 0);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 0);
    GetData(tensor2, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 1);
    EXPECT_EQ(Internal::g_bufId, 0xF);

    tbufPool.Reset();
    EXPECT_EQ(tbufPool.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(tbufPool.tBufPoolImpl.maxAddr_, 0);
}

class CustomTBufpoolTest : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

class MyBufPool {
public:
    __aicore__ inline MyBufPool() {
        Init();
    }

    template<class T> 
    __aicore__ inline bool InitBuffer(T& que) {
        auto curPoolAddr = this->GetCurAddr();

        // call internal func to initnitial bufhandle
        que.InitStartBufHandle(this->GetBufHandle(0), BUF_ID_SIZE, BUFFER_LENGTH);
        for (int32_t i = 0; i < BUF_ID_SIZE; i++) {
            que.InitBufHandle(this, i, this->GetBufHandle(i), curPoolAddr + i * BUFFER_LENGTH, BUFFER_LENGTH);
        }

        this->SetCurBufSize(BUF_ID_SIZE);

        return true;
    }

    template<AscendC::TPosition bufPos>
    __aicore__ inline bool InitBuffer(AscendC::TBuf<bufPos>& buf, uint32_t len) {
        auto ptr = this->GetBufHandle(this->GetCurBufSize());
        auto curPoolAddr = this->GetCurAddr();

        // call internal func to initnitial bufhandle
        buf.InitStartBufHandle(ptr, 1, len);

        this->SetCurBufSize(BUF_ID_SIZE);
        return true;
    }
    EXTERN_IMPL_BUFPOOL(MyBufPool, AscendC::TPosition::VECCALC, BUF_ID_SIZE);
};

TEST_F(CustomTBufpoolTest, testCirculaTBufpool)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1> tmpQue;
    TQue<TPosition::VECIN, 1> inQueX;
    TQue<TPosition::VECIN, 1> inQueY;
    TQue<TPosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> tmpBuf;
    MyBufPool myPool;
    pipe.InitBuffer(tmpQue, 2, BUFFER_LENGTH);
    pipe.InitBufPool(myPool, BUF_ID_SIZE * BUFFER_LENGTH);
    myPool.InitBuffer(inQueX);
    myPool.InitBuffer(inQueY);
    myPool.InitBuffer(outQue);
    myPool.InitBuffer(tmpBuf, BUFFER_LENGTH);

    EXPECT_EQ(myPool.tBufPoolImpl.curBufSize_, BUF_ID_SIZE);
    EXPECT_EQ(myPool.tBufPoolImpl.startAddr_, 2 * BUFFER_LENGTH);
    EXPECT_EQ(myPool.tBufPoolImpl.maxAddr_, 2 * BUFFER_LENGTH);

    LocalTensor<float> inX = inQueX.AllocTensor<float>();
    inQueX.EnQue(inX);
    inX = inQueX.DeQue<float>();

    uint32_t actualAddr;
    uint32_t actualLen;
    uint8_t actualBufId;
    GetData(inX, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 2 * BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 2);

    LocalTensor<float> inY = inQueY.AllocTensor<float>();
    inQueY.EnQue(inY);
    inY = inQueY.DeQue<float>();

    GetData(inY, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 3 * BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 3);

    LocalTensor<float> outTensor = outQue.AllocTensor<float>();
    outQue.EnQue(outTensor);
    outTensor = outQue.DeQue<float>();

    GetData(outTensor, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 4 * BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 4);

    inQueX.FreeTensor(inX);
    inX = inQueX.AllocTensor<float>();
    inQueX.EnQue(inX);
    inX = inQueX.DeQue<float>();

    inQueY.FreeTensor(inY);
    outQue.FreeTensor(outTensor);

    GetData(inX, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 5 * BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);
    EXPECT_EQ(actualBufId, 5);

    inQueX.FreeTensor(inX);
    LocalTensor<float> tmp = tmpBuf.Get<float>();

    GetData(tmp, actualAddr, actualLen, actualBufId);
    EXPECT_EQ(actualAddr, 2 * BUFFER_LENGTH);
    EXPECT_EQ(actualLen, BUFFER_LENGTH);

    myPool.Reset();
    EXPECT_EQ(myPool.tBufPoolImpl.curBufSize_, 0);
    EXPECT_EQ(myPool.tBufPoolImpl.maxAddr_, 2 * BUFFER_LENGTH);
}

} // namespace AscendC