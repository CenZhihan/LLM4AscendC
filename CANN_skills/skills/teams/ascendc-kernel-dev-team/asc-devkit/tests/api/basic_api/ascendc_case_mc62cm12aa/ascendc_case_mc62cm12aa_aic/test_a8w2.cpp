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

namespace AscendC {
template <class A_TYPE, class C_TYPE> class E2eCase {
public:
    __aicore__ inline E2eCase() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR cGM, int32_t m, int32_t k)
    {
        mLength = m;
        kLength = k;

        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE *>(aGM), m * k);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE *>(cGM), m * k);

        if constexpr(IsSameType<C_TYPE, int4b_t>::value) {
            pipe.InitBuffer(queL1, 1, m * k / 2);
            pipe.InitBuffer(queUB, 1, m * k / 2);
        } else if constexpr(IsSameType<C_TYPE, int8_t>::value) {
            pipe.InitBuffer(queL1, 1, m * k);
            pipe.InitBuffer(queUB, 1, m * k);
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<C_TYPE> tmpL1 = queL1.AllocTensor<C_TYPE>();
        LocalTensor<C_TYPE> tmpUB = queUB.AllocTensor<C_TYPE>();
        uint8_t coeff = 1;
        if constexpr(IsSameType<C_TYPE, int4b_t>::value) {
            coeff = 2;
        }

        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(mLength, 16);
        loadDataParams.kStep = DivCeil(kLength / coeff, 32);
        loadDataParams.srcStride = DivCeil(mLength, 16);
        loadDataParams.dstStride = DivCeil(mLength, 16);

        Nd2NzParamsV2 nz2ndParams;
        nz2ndParams.lookupTable0 = 258;
        nz2ndParams.lookupTable1 = 1027;

        LoadData(tmpL1, aGlobal_, loadDataParams, nz2ndParams);

        pipe_barrier(PIPE_ALL);
        DataCopy(tmpUB, tmpL1, mLength * kLength);
        pipe_barrier(PIPE_ALL);
        DataCopy(cGlobal_, tmpUB, mLength * kLength);
        pipe_barrier(PIPE_ALL);
    }

private:
    TPipe pipe;
    int32_t mLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    TQue<TPosition::A1, 1> queL1;
    TQue<TPosition::VECOUT, 1> queUB;

    GlobalTensor<A_TYPE> aGlobal_;
    GlobalTensor<C_TYPE> cGlobal_;
};

template <typename AType, typename BType>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR cGM, uint16_t m, uint16_t k)
{
    if (block_idx >= 1) {
        return;
    }

    E2eCase<AType, BType> ins;
    ins.Init(aGM, cGM, m, k);
    ins.Process();
    return;
}

struct A8W2TestParams {
    uint16_t m;
    uint16_t k;
    void (*cal_func)(uint8_t*, uint8_t*, uint16_t, uint16_t);
};

class A8W2TestSuite
    : public testing::Test
    , public testing::WithParamInterface<A8W2TestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(
    TEST_LOAD_DATA, A8W2TestSuite,
    ::testing::Values(
        A8W2TestParams{ 16, 16, E2eKernel<int2b_t, int4b_t>},
        A8W2TestParams{ 64, 128, E2eKernel<int2b_t, int4b_t>},
        A8W2TestParams{ 64, 128, E2eKernel<int2b_t, int8_t>},
        A8W2TestParams{ 64, 128, E2eKernel<int4b_t, int8_t>}
    ));

TEST_P(A8W2TestSuite, A8W2TestCase)
{
    auto param = GetParam();
    uint8_t* dst_gm = new uint8_t[param.m * param.k];
    uint8_t* src0_gm = new uint8_t[param.m * param.k];
    param.cal_func(dst_gm, src0_gm, param.m, param.k);
    delete [] dst_gm;
    delete [] src0_gm;
}
}  // namespace AscendC
