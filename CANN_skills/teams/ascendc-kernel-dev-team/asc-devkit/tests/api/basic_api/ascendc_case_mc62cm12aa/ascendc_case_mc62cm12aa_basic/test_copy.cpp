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
#include "test_utils.h"

using namespace std;
using namespace AscendC;

template <typename T, typename castType, bool isMaskCounterMode, uint32_t maskSize>
class KernelCopyCounter {
public:
    __aicore__ inline KernelCopyCounter() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t dataSizeIn, uint32_t counterIn, uint32_t isBitMapIn)
    {
        dataSize = dataSizeIn;
        counter = counterIn;
        isBitMap = isBitMapIn;
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dataSize);
        pipe.InitBuffer(inQueue, 1, dataSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        if constexpr (isMaskCounterMode) {
            ComputeMaskCounter();
        } else {
            if constexpr (maskSize == 0) {
                ComputeMask0();
            } else if constexpr (maskSize == 1) {
                ComputeMask1();
            }else if constexpr (maskSize == 4) {
                ComputeMask4();
            }
        }
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();

        GlobalTensor<castType> srcCastGlobal = srcGlobal.template ReinterpretCast<castType>();
        LocalTensor<castType> srcCastLocal = srcLocal.template ReinterpretCast<castType>();

        if constexpr (SupportType<castType, int4b_t, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            DataCopy(srcCastLocal, srcCastGlobal, srcCastGlobal.GetSize());
        } else {
            DataCopy(srcCastLocal, srcCastGlobal, srcCastGlobal.GetSize());
        }

        inQueue.EnQue<T>(srcLocal);
    }

    __aicore__ inline void ComputeMaskCounter()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();

        SetMaskCount();
    
        T zero = 0;
        Duplicate(dstLocal, zero, dataSize);
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint32_t repeatTimes = (dataSize + repeatElm - 1) / repeatElm;
        if(isBitMap){
            SetVectorMask<T, MaskMode::COUNTER>(0, counter);
            pipe_barrier(PIPE_ALL); 
            Copy<T, false>(dstLocal, srcLocal, MASK_PLACEHOLDER, repeatTimes, params);
            pipe_barrier(PIPE_ALL);
        }else{
            SetVectorMask<T, MaskMode::COUNTER>(counter);
            pipe_barrier(PIPE_ALL); 
            Copy<T, false>(dstLocal, srcLocal, MASK_PLACEHOLDER, repeatTimes, params);
            pipe_barrier(PIPE_ALL);
        }

        SetMaskNorm();
        ResetMask();
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void ComputeMask0()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        if constexpr (sizeof(castType) == 1) {
            Duplicate((LocalTensor<uint8_t> &)dstLocal, (uint8_t)0, dataSize);
        } else {
            Duplicate(dstLocal, (castType)0, dataSize);
        }
        Copy(dstLocal, srcLocal, counter);
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void ComputeMask1()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        if constexpr (sizeof(castType) == 1) {
            Duplicate((LocalTensor<uint8_t> &)dstLocal, (uint8_t)0, dataSize);
        } else {
            Duplicate(dstLocal, (castType)0, dataSize);
        }
        uint64_t mask = dataSize % 256;
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint32_t repeatTimes = (dataSize + repeatElm - 1) / repeatElm;                                                                                                                                                                                                                                               \
        Copy(dstLocal, srcLocal, mask, repeatTimes, params);              
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

        __aicore__ inline void ComputeMask4()
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        if constexpr (sizeof(castType) == 1) {
            Duplicate((LocalTensor<uint8_t> &)dstLocal, (uint8_t)0, dataSize);
        } else {
            Duplicate(dstLocal, (castType)0, dataSize);
        }
        uint64_t mask[4] = {0xffffffff, 0, 0, 0};
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint32_t repeatTimes = (dataSize + repeatElm - 1) / repeatElm;                                                                                                                                                                                                                                               \
        Copy(dstLocal, srcLocal, mask, repeatTimes, params);              
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize;
    CopyRepeatParams params = {1, 1, 8, 8};
    uint32_t counter, isBitMap;
};

template <typename T, typename castType, bool isMaskCounterMode, uint32_t maskSize>
void main_KernelCopyCounter_test(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t dataSizeIn, uint32_t counterIn, uint32_t isBitMapIn)
{
    KernelCopyCounter<T, castType, isMaskCounterMode, maskSize> op;
    op.Init(srcGm, dstGm, dataSizeIn, counterIn, isBitMapIn);
    op.Process(); 
}

struct KernelCopyCounterTestParams {
    uint32_t dataSize;
    uint32_t counter;
    uint32_t isBitMapIn;
    void (*cal_func) (GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t);
};

class KernelCopyCounterTestSuite : public testing::Test, public testing::WithParamInterface<KernelCopyCounterTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KernelCopyCounterTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "KernelCopyCounterTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_KernelCopyCounter, KernelCopyCounterTestSuite,
    ::testing::Values(
    KernelCopyCounterTestParams { 32,     10,    1,    main_KernelCopyCounter_test<uint32_t  , half,     true,     0>         },
    KernelCopyCounterTestParams { 64,     15,    0,    main_KernelCopyCounter_test<uint32_t  , bfloat16_t,     true,     0>       },
    KernelCopyCounterTestParams { 128,    128,    1,    main_KernelCopyCounter_test<int32_t   , int4b_t,     true,     0>          },
    KernelCopyCounterTestParams { 160,    128,    1,    main_KernelCopyCounter_test<int32_t   , fp4x2_e2m1_t,     true,     0>    },
    KernelCopyCounterTestParams { 256,    128,    0,    main_KernelCopyCounter_test<float     , fp4x2_e1m2_t,     true,     0>    },

    KernelCopyCounterTestParams { 32,     10,    1,    main_KernelCopyCounter_test<uint32_t  , hifloat8_t,     true,     0>       },
    KernelCopyCounterTestParams { 64,     15,    1,    main_KernelCopyCounter_test<uint32_t  , fp8_e5m2_t,     true,     0>    },
    KernelCopyCounterTestParams { 128,    128,    1,    main_KernelCopyCounter_test<int32_t   , fp8_e4m3fn_t,     true,     0>    },
    KernelCopyCounterTestParams { 160,    128,    0,    main_KernelCopyCounter_test<int32_t   , fp8_e8m0_t,     true,     0>    },
    KernelCopyCounterTestParams { 256,    128,    1,    main_KernelCopyCounter_test<float     , int64_t,     true,     0>          },

    KernelCopyCounterTestParams { 64,     10,    1,    main_KernelCopyCounter_test<uint16_t  , half,     true,     0>             },
    KernelCopyCounterTestParams { 128,    128,    1,    main_KernelCopyCounter_test<uint16_t  , bfloat16_t,     true,     0>       },
    KernelCopyCounterTestParams { 160,    128,    1,    main_KernelCopyCounter_test<int16_t   , int4b_t,     true,     0>          },
    KernelCopyCounterTestParams { 256,    128,    1,    main_KernelCopyCounter_test<int16_t   , fp4x2_e2m1_t,     true,     0>    },
    KernelCopyCounterTestParams { 512,    128,    1,    main_KernelCopyCounter_test<half      , fp4x2_e1m2_t,     true,     0>    },

    KernelCopyCounterTestParams { 64,     10,    1,    main_KernelCopyCounter_test<uint16_t  , hifloat8_t,     true,     0>       },
    KernelCopyCounterTestParams { 128,    128,    1,    main_KernelCopyCounter_test<uint16_t  , fp8_e5m2_t,     true,     0>    },
    KernelCopyCounterTestParams { 160,    128,    1,    main_KernelCopyCounter_test<int16_t   , fp8_e4m3fn_t,     true,     0>    },
    KernelCopyCounterTestParams { 256,    128,    1,    main_KernelCopyCounter_test<int16_t   , fp8_e8m0_t,     true,     0>    },
    KernelCopyCounterTestParams { 512,    128,    1,    main_KernelCopyCounter_test<half      , int64_t,     true,     0>          },

    KernelCopyCounterTestParams { 160,    128,    1,    main_KernelCopyCounter_test<int16_t,     int64_t,     true,     0>         },
    KernelCopyCounterTestParams { 160,    128,    1,    main_KernelCopyCounter_test<uint32_t,    int64_t,     true,     0>         },
    

    KernelCopyCounterTestParams { 64,     62,    0,     main_KernelCopyCounter_test<uint8_t  , uint8_t,   false,    0>       },
    KernelCopyCounterTestParams { 128,    128,    0,     main_KernelCopyCounter_test<int8_t  , int8_t,   false,    0>    },
    KernelCopyCounterTestParams { 160,    128,    0,     main_KernelCopyCounter_test<hifloat8_t   , hifloat8_t,   false,    0>    },
    KernelCopyCounterTestParams { 256,    128,    0,     main_KernelCopyCounter_test<fp8_e4m3fn_t   , fp8_e4m3fn_t,   false,    0>    },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<fp8_e5m2_t      , fp8_e5m2_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<uint16_t      , uint16_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<int16_t      , int16_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<bfloat16_t      , bfloat16_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<half      , half,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<uint32_t      , uint32_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<int32_t      , int32_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<float      , float,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<uint64_t      , uint64_t,   false,    0>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<int64_t      , int64_t,   false,    0>          },

    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<uint16_t      , uint16_t,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<int16_t      , int16_t,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<bfloat16_t      , bfloat16_t,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<half      , half,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<uint32_t      , uint32_t,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<int32_t      , int32_t,   false,    1>          },
    KernelCopyCounterTestParams { 512,    128,    0,     main_KernelCopyCounter_test<float      , float,   false,    1>          },

    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<uint16_t      , uint16_t,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<int16_t      , int16_t,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<bfloat16_t      , bfloat16_t,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<half      , half,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<uint32_t      , uint32_t,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<int32_t      , int32_t,   false,    4>          },
    KernelCopyCounterTestParams { 512,    128,    0,    main_KernelCopyCounter_test<float      , float,   false,    4>          }
    ));

TEST_P(KernelCopyCounterTestSuite, KernelCopyCounterTestCase)
{
    auto param = GetParam();

    uint32_t dataSize = param.dataSize;

    uint8_t *inputGm = new uint8_t[dataSize * 4 * 10] { 0x00 };
    uint8_t *outputGm= new uint8_t[dataSize * 4 * 10] { 0x00 };

    param.cal_func(inputGm, outputGm, param.dataSize, param.counter, param.isBitMapIn);

    for (int32_t i = 0; i < dataSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
    delete [] inputGm;
    delete [] outputGm;
}