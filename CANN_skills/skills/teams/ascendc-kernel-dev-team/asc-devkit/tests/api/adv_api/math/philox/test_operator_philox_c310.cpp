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
 
 template <uint16_t Rounds, typename srcType> class KernelPhilox {
 public:
     __aicore__ inline KernelPhilox()
     {}
     __aicore__ inline void Init(GM_ADDR dstGm, uint32_t dstSize)
     {
        count = dstSize;
        const int align = 32 / sizeof(srcType);
        alignSize = (count + align - 1) / align * align;
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), alignSize);
        pipe.InitBuffer(outQueue, 1, alignSize * sizeof(srcType));
     }
     __aicore__ inline void Process(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint32_t seed4,
        uint32_t seed5)
     {
        Compute(seed0, seed1, seed2, seed3, seed4, seed5);
        CopyOut();
     }
 
 private:
     __aicore__ inline void Compute(uint32_t seed0, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint32_t seed4,
        uint32_t seed5)
     {
        LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        PhiloxRandom<Rounds>(dstLocal, { seed0, seed1 }, { seed2, seed3, seed4, seed5}, count);
        outQueue.EnQue<srcType>(dstLocal);
     }
     __aicore__ inline void CopyOut()
     {
         LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
         DataCopy(dstGlobal, dstLocal, alignSize);
         outQueue.FreeTensor(dstLocal);
     }
 
 private:
     GlobalTensor<srcType> dstGlobal;
 
     TPipe pipe;
     TQue<TPosition::VECOUT, 1> outQueue;
     uint32_t count;
     uint32_t alignSize;
 };
 
 template <uint16_t Rounds, typename srcType>
 __global__ __aicore__ void MainKernelPhilox(uint8_t *dstGm, uint32_t dstSize, const uint32_t s[6])
 {
    KernelPhilox<Rounds, srcType> op;
    op.Init(dstGm, dstSize);
    op.Process(s[0], s[1], s[2], s[3], s[4], s[5]);
 }
 
 struct PhiloxTestParams {
     void (*cal_func)(uint8_t *, uint32_t, const uint32_t *);
     uint32_t dstSize;
     uint32_t s[6];
 };
 
class PhiloxSimpleTestsuite : public testing::TestWithParam<PhiloxTestParams> {};
 
 INSTANTIATE_TEST_CASE_P(PhiloxSimpleTestCase, PhiloxSimpleTestsuite,
     ::testing::Values(
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, uint32_t>, 1025, {0, 0, 0, 0, 0, 0} },

        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, int32_t>, 1025, {0, 0, 0, 0, 0, 0} },

        PhiloxTestParams { MainKernelPhilox<7, float>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<7, float>, 1025, {0, 0, 0, 0, 0, 0} },

        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, uint32_t>, 1025, {0, 0, 0, 0, 0, 0} },

        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, int32_t>, 1025, {0, 0, 0, 0, 0, 0} },

        PhiloxTestParams { MainKernelPhilox<10, float>, 1, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 4, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 63, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 64, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 255, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 256, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 257, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 1024, {0, 0, 0, 0, 0, 0} },
        PhiloxTestParams { MainKernelPhilox<10, float>, 1025, {0, 0, 0, 0, 0, 0} }
     )
 );
 
 TEST_P(PhiloxSimpleTestsuite, PhiloxSimpleTestCase)
 {
     auto param = GetParam();
     uint8_t dstGm[param.dstSize * 4] = {0};
     param.cal_func(dstGm, param.dstSize, param.s);
     for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
         EXPECT_EQ(dstGm[i], 0x00);
     }
 }
 