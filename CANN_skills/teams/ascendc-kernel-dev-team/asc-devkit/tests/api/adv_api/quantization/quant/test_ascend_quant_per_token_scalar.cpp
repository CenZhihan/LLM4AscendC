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
 #include "kernel_operator.h"
 #include "kernel_utils.h"
 #include <math.h>
 #include <iostream>
 #include <fstream>

 using namespace std;
 using namespace AscendC;

 namespace AscendC {
 template <typename srcType, typename dstType, typename scaleType, bool has_offset, bool has_workspace>
 class KernelAscendQuantPerTokenScalar {
 public:
     __aicore__ inline KernelAscendQuantPerTokenScalar() {}
     __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR scale_gm,
                                 uint32_t m_num, uint32_t n_num, uint32_t calCount_num)
     {
         m = m_num;
         n = n_num;
         calCount = calCount_num;
         uint32_t align_size = 32;
         src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), m*n);
         dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ dstType *>(dst_gm), m*n);
         scale_global.SetGlobalBuffer(reinterpret_cast<__gm__ scaleType *>(scale_gm), align_size);

         pipe.InitBuffer(inQueue, 1, m * n * sizeof(srcType));
         pipe.InitBuffer(inQueueScale, 1, align_size * sizeof(scaleType));
         pipe.InitBuffer(outQueue, 1, m * n * sizeof(dstType));
     }
     __aicore__ inline void Process()
     {
         CopyIn();
         Compute();
         CopyOut();
     }

 private:
     __aicore__ inline void CopyIn()
     {
         LocalTensor<srcType> srcLocal = inQueue.AllocTensor<srcType>();
         LocalTensor<scaleType> scaleLocal = inQueueScale.AllocTensor<scaleType>();

         DataCopy(srcLocal, src_global, m*n);
         DataCopy(scaleLocal, scale_global, 16);

         inQueue.EnQue(srcLocal);
         inQueueScale.EnQue(scaleLocal);
     }
     __aicore__ inline void Compute()
     {
         LocalTensor<dstType> dstLocal = outQueue.AllocTensor<dstType>();

         LocalTensor<srcType> srcLocal = inQueue.DeQue<srcType>();
         LocalTensor<scaleType> scaleLocal = inQueueScale.DeQue<scaleType>();

         constexpr static bool isReuseSource = false;
         constexpr static AscendQuantConfig config = {has_offset, -1};
         constexpr static AscendQuantPolicy policy = AscendQuantPolicy::PER_TOKEN;
         AscendQuantParam para;
         para.m = m;
         para.n = n;
         para.calCount = calCount;
         scaleType offset = 2;
         if constexpr(has_workspace) {
             LocalTensor<uint8_t> sharedTmpBuffer;
             bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
             AscendQuant<dstType, srcType, scaleType, isReuseSource, config, policy>(dstLocal, srcLocal, sharedTmpBuffer, scaleLocal, offset, para);
         } else {
             AscendQuant<dstType, srcType, scaleType, isReuseSource, config, policy>(dstLocal, srcLocal, scaleLocal, offset, para);
         }
         outQueue.EnQue<dstType>(dstLocal);
         inQueue.FreeTensor(srcLocal);
         inQueueScale.FreeTensor(scaleLocal);
     }
     __aicore__ inline void CopyOut()
     {
         LocalTensor<dstType> dstLocal = outQueue.DeQue<dstType>();
         DataCopy(dst_global, dstLocal, m * n);
         outQueue.FreeTensor(dstLocal);
     }

 private:
     GlobalTensor<srcType> src_global;
     GlobalTensor<scaleType> scale_global;
     GlobalTensor<scaleType> offset_global;
     GlobalTensor<dstType> dst_global;

     TPipe pipe;
     TQue<TPosition::VECIN, 1> inQueue, inQueueScale;
     TQue<TPosition::VECOUT, 1> outQueue;

     uint32_t m;
     uint32_t n;
     uint32_t calCount;
 };
 } // namespace AscendC

 template <typename srcType, typename dstType, typename scaleType, bool has_offset, bool has_workspace>
 __global__ __aicore__ void Entry(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* scaleGm,
                                  uint32_t m, uint32_t n, uint32_t calCount)
 {
     AscendC::KernelAscendQuantPerTokenScalar<srcType, dstType, scaleType, has_offset, has_workspace> op;
     op.Init(srcGm, dstGm, scaleGm, m, n, calCount);
     op.Process();
 }

 struct AscendQuantPerTokenScalarTestParams {
     uint32_t m;
     uint32_t n;
     uint32_t calCount;
     void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
 };

 class AscendQuantPerTokenScalarTestsuite : public testing::Test, public testing::WithParamInterface<AscendQuantPerTokenScalarTestParams> {
 protected:
     void SetUp()
     {
         AscendC::SetGCoreType(2);
     }

     void TearDown()
     {
         AscendC::SetGCoreType(0);
     }
 };

 INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ASCEND_QUANT_PER_TOKEN, AscendQuantPerTokenScalarTestsuite,
     ::testing::Values(
        //  AscendQuantPerTokenScalarTestParams{ 10, 256, 1280, Entry<half, fp4x2_e1m2_t, half, false, true> },
        //  AscendQuantPerTokenScalarTestParams{ 10, 256, 1280, Entry<half, fp4x2_e2m1_t, half, true, false> },

         AscendQuantPerTokenScalarTestParams{ 10, 256, 1280, Entry<half, fp8_e5m2_t, half, false, true> },
         AscendQuantPerTokenScalarTestParams{ 10, 288, 1152, Entry<half, fp8_e4m3fn_t, half, true, false> },
         AscendQuantPerTokenScalarTestParams{ 7, 256, 512, Entry<bfloat16_t, fp8_e5m2_t, bfloat16_t, false, true> },
         AscendQuantPerTokenScalarTestParams{ 7, 288, 864, Entry<bfloat16_t, fp8_e4m3fn_t, bfloat16_t, true, false> },
         AscendQuantPerTokenScalarTestParams{ 10, 256, 1280, Entry<float, fp8_e5m2_t, float, false, true> },
         AscendQuantPerTokenScalarTestParams{ 10, 288, 1152, Entry<float, fp8_e4m3fn_t, float, true, false> },
         AscendQuantPerTokenScalarTestParams{ 7, 256, 512, Entry<half, fp8_e5m2_t, float, false, true> },
         AscendQuantPerTokenScalarTestParams{ 7, 288, 864, Entry<half, fp8_e4m3fn_t, float, true, false> },
         AscendQuantPerTokenScalarTestParams{ 7, 256, 512, Entry<bfloat16_t, fp8_e5m2_t, float, false, true> },
         AscendQuantPerTokenScalarTestParams{ 7, 288, 864, Entry<bfloat16_t, fp8_e4m3fn_t, float, true, false> },

         AscendQuantPerTokenScalarTestParams{ 10, 288, 1152, Entry<half, hifloat8_t, half, true, false> },
         AscendQuantPerTokenScalarTestParams{ 7, 288, 864, Entry<bfloat16_t, hifloat8_t, bfloat16_t, true, false> },
         AscendQuantPerTokenScalarTestParams{ 6, 288, 1440, Entry<float, hifloat8_t, float, true, true> },
         AscendQuantPerTokenScalarTestParams{ 8, 512, 3584, Entry<half, hifloat8_t, float, true, false> },
         AscendQuantPerTokenScalarTestParams{ 4, 256, 768, Entry<bfloat16_t, hifloat8_t, float, true, true> },

         AscendQuantPerTokenScalarTestParams{ 10, 288, 1152, Entry<half, int8_t, half, true, false> },
         AscendQuantPerTokenScalarTestParams{ 7, 288, 864, Entry<bfloat16_t, int8_t, bfloat16_t, true, false> },
         AscendQuantPerTokenScalarTestParams{ 6, 288, 1440, Entry<float, int8_t, float, true, true> },
         AscendQuantPerTokenScalarTestParams{ 8, 512, 3584, Entry<half, int8_t, float, true, false> },
         AscendQuantPerTokenScalarTestParams{ 4, 256, 768, Entry<bfloat16_t, int8_t, float, true, true> }
     ));

 TEST_P(AscendQuantPerTokenScalarTestsuite, AscendQuantPerTokenScalarOpTestCase)
 {
     const int dataBytes = 4;
     auto param = GetParam();
     uint8_t srcGm[param.m * param.n * dataBytes];
     uint8_t dstGm[param.m * param.n * dataBytes];
     uint8_t scaleGm[param.m * dataBytes];
     param.cal_func(srcGm, dstGm, scaleGm, param.m, param.n, param.calCount);
 }
