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
#include "kernel_utils.h"
using namespace std;
using namespace AscendC;
enum TestMode {
    CAL_MODE,
    BUF_CAL_MODE
};
template<typename DstT, typename SrcT, int32_t calCount, int32_t dataSize, TestMode mode>
class KernelIsInf {
public:
    __aicore__ inline KernelIsInf()
    {}
    __aicore__ inline void Init(GM_ADDR dst_gm, GM_ADDR src_gm)
    {
        srcSize = dataSize;
        dstSize = dataSize;
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT *>(src_gm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT *>(dst_gm), dstSize);

        pipe.InitBuffer(inQueue, 1, dataSize * sizeof(SrcT));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(DstT));

        mask = calCount;
    }
    __aicore__ inline void Process()
    {
        AscendCUtils::SetOverflow(1);
        CopyIn();
        Compute();
        CopyOut();
        AscendCUtils::SetOverflow(0);
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<SrcT> srcLocal = inQueue.AllocTensor<SrcT>();
        DataCopy(srcLocal, src_global, srcSize);
        inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<SrcT> srcLocal = inQueue.DeQue<SrcT>();
        if constexpr (mode == CAL_MODE) {
            if constexpr (std::is_same_v<DstT, bool>) {
                LocalTensor<DstT> dstLocal = outQueue.AllocTensor<DstT>();
                Duplicate((LocalTensor<int8_t> &)dstLocal, (int8_t)0, dataSize);
                IsInf(dstLocal, srcLocal, calCount);
                outQueue.EnQue(dstLocal);
            } else {
                LocalTensor<DstT> dstLocal = outQueue.AllocTensor<DstT>();
                Duplicate(dstLocal, (DstT)0, dataSize);
                IsInf(dstLocal, srcLocal, calCount);
                outQueue.EnQue(dstLocal);
            }
        } else {
            TBuf<TPosition::VECCALC> tbuf;
            if constexpr (sizeof(SrcT) == sizeof(float)) {
                pipe.InitBuffer(tbuf, dataSize * sizeof(float));
            } else {
                pipe.InitBuffer(tbuf, dataSize * sizeof(float) * 4);
            }
            LocalTensor<uint8_t> tmpLocal = tbuf.Get<uint8_t>();
            if constexpr (std::is_same_v<DstT, bool>) {
                LocalTensor<DstT> dstLocal = outQueue.AllocTensor<DstT>();
                Duplicate((LocalTensor<int8_t> &)dstLocal, (int8_t)0, dataSize);
                IsInf(dstLocal, srcLocal, tmpLocal, calCount);
                outQueue.EnQue(dstLocal);
            } else {
                LocalTensor<DstT> dstLocal = outQueue.AllocTensor<DstT>();
                Duplicate(dstLocal, (DstT)0, dataSize);
                IsInf(dstLocal, srcLocal, tmpLocal, calCount);
                outQueue.EnQue(dstLocal);
            }
        }
        inQueue.FreeTensor(srcLocal);

    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DstT> dstLocal = outQueue.DeQue<DstT>();
        DataCopy(dst_global, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<SrcT> src_global;
    GlobalTensor<DstT> dst_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t srcSize = 0;
    uint32_t dstSize = 0;
    uint32_t mask;
};

struct IsInfParams {
    void (*CalFunc)();
};

template<typename DstT, typename SrcT, int32_t calCount, int32_t dataSize, TestMode mode>
void RunCase() {
    int src_byte_size = sizeof(SrcT);
    int dst_byte_size = sizeof(DstT);

    uint8_t dstGm[calCount * dst_byte_size] = {0};
    uint8_t srcGm[calCount * src_byte_size] = {0};

    KernelIsInf<DstT, SrcT, calCount, dataSize, mode> op;
    op.Init(dstGm, srcGm);
    op.Process();
}

class IsInfTestsuite : public testing::TestWithParam<IsInfParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ADVANCE_API_ISINF, IsInfTestsuite,
    ::testing::Values(
        IsInfParams { RunCase<float, float, 1024, 1024, CAL_MODE> },
        IsInfParams { RunCase<float, float, 1023, 1024, CAL_MODE> },
        IsInfParams { RunCase<float, float, 512, 1024, CAL_MODE> },
        IsInfParams { RunCase<float, float, 1, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1024, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1023, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, float, 512, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1, 1024, CAL_MODE> },
        IsInfParams { RunCase<half, half, 1024, 1024, CAL_MODE> },
        IsInfParams { RunCase<half, half, 1023, 1024, CAL_MODE> },
        IsInfParams { RunCase<half, half, 512, 1024, CAL_MODE> },
        IsInfParams { RunCase<half, half, 1, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1024, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1023, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, half, 512, 1024, CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1, 1024, CAL_MODE> },
        
        IsInfParams { RunCase<float, float, 1024, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<float, float, 1023, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<float, float, 512, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<float, float, 1, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1024, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1023, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, float, 512, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, float, 1, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<half, half, 1024, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<half, half, 1023, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<half, half, 512, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<half, half, 1, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1024, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1023, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, half, 512, 1024, BUF_CAL_MODE> },
        IsInfParams { RunCase<bool, half, 1, 1024, BUF_CAL_MODE> }
    )
);

TEST_P(IsInfTestsuite, IsInfTestCase)
{
    auto param = GetParam();
    param.CalFunc();
}
