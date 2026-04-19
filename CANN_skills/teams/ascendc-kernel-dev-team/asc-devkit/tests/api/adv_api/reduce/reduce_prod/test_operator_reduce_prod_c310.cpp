/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gtest/gtest.h"
#define private public
#define protected public
#include "kernel_operator.h"

namespace AscendC {
template <typename T, typename pattern, bool IsReuseSource>
class KernelReduceProd {
 public:
  __aicore__ inline KernelReduceProd() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const uint32_t heightIn, const uint32_t widthIn, bool srcInnerPadIn) {
    this->height = heightIn;
    this->width = widthIn;
    this->srcInnerPad = srcInnerPadIn;
    if constexpr (std::is_same_v<pattern, AscendC::Pattern::Reduce::AR>) {
      this->outputWidth = heightIn;
    } else {
      this->outputWidth = widthIn;
    }
    const uint32_t widthAlginSize = 32 / sizeof(T);
    this->outputWidthAlign = (outputWidth + widthAlginSize - 1) / widthAlginSize * widthAlginSize;

    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ T*)x, heightIn * widthIn);
    yGm.SetGlobalBuffer((__gm__ T*)y, outputWidthAlign);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(outQueueY, 1, outputWidthAlign * sizeof(T));
    pipe.InitBuffer(inQueueX, 1, heightIn * widthIn * sizeof(T));
  }

  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

  __aicore__ inline void Compute() {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    uint32_t srcShape[2] = {height, width};
    ReduceProd<T, pattern, IsReuseSource>(yLocal, xLocal, srcShape, srcInnerPad);

    outQueueY.EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }

 private:
  __aicore__ inline void CopyIn() {
    // alloc tensor from queue memory
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams;
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = height * width * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;
    padParams.rightPadding = 0;
    DataCopyPad(xLocal, xGm, copyParams, padParams);

    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void CopyOut() {
    // deque output tensor from VECOUT queue
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyPad(yGm, yLocal, {1, static_cast<uint32_t>(outputWidth * sizeof(T)), 0, 0});
    // free output tensor for reuse
    outQueueY.FreeTensor(yLocal);
  }

 private:
  TPipe pipe;
  // create queue for output, in this case depth is equal to buffer num
  TQue<TPosition::VECOUT, 1> outQueueY;
  // create queues for input, in this case depth is equal to buffer num
  TQue<TPosition::VECIN, 1> inQueueX;

  GlobalTensor<T> xGm;
  GlobalTensor<T> yGm;

  uint32_t height;
  uint32_t width;
  uint32_t outputWidthAlign;
  uint32_t outputWidth;
  bool srcInnerPad;
};
}

template <typename T, uint32_t height, uint32_t width, typename pattern, bool isReuseSource, bool srcInnerPad>
__aicore__ inline void MainReduceProdTest(uint8_t* x, uint8_t* y)
{
    AscendC::KernelReduceProd<T, pattern, isReuseSource> op;
    op.Init(x, y, height, width, srcInnerPad);
    op.Process();
}

struct ReduceProdTestParams {
    void (*cal_func)(uint8_t *, uint8_t *);
};

class ReduceProdTestsuite : public testing::Test, public testing::WithParamInterface<ReduceProdTestParams> {};

INSTANTIATE_TEST_CASE_P(TEST_ReduceProd, ReduceProdTestsuite,
    ::testing::Values(
      ReduceProdTestParams{ MainReduceProdTest<float, 1, 128, AscendC::Pattern::Reduce::AR, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 5, 120, AscendC::Pattern::Reduce::AR, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 15, 56, AscendC::Pattern::Reduce::AR, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 31, 128, AscendC::Pattern::Reduce::AR, false, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 32, 120, AscendC::Pattern::Reduce::AR, false, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 64, 56, AscendC::Pattern::Reduce::AR, false, true> },

      ReduceProdTestParams{ MainReduceProdTest<float, 1, 64, AscendC::Pattern::Reduce::RA, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 2, 120, AscendC::Pattern::Reduce::RA, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 12, 56, AscendC::Pattern::Reduce::RA, true, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 18, 64, AscendC::Pattern::Reduce::RA, false, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 63, 120, AscendC::Pattern::Reduce::RA, false, true> },
      ReduceProdTestParams{ MainReduceProdTest<float, 74, 56, AscendC::Pattern::Reduce::RA, false, true> }
    ));

TEST_P(ReduceProdTestsuite, ReduceProdTestCase)
{
    auto param = GetParam();
    uint8_t x[40960] = {0};
    uint8_t y[40960] = {0};

    param.cal_func(x, y);
    for (int32_t i = 0; i < (sizeof(y) / sizeof(y[0])); i++) {
        EXPECT_EQ(y[i], 0x00);
    }
}
