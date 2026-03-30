
#include "kernel_operator.h"
#define __NPU_TILING__
#include "leaky_relu_custom_tiling_data.h"

// 参考 CANN Ascend C API: 4.1.3.7 LeakyRelu（逐元素，对负值乘以scalar，非负值保持不变）

constexpr int32_t BUFFER_NUM = 2; // 每个队列的本地tensor个数

using DTYPE = float;

class KernelLeakyRelu {
public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, float negativeSlope)
    {
        this->negSlope = negativeSlope;
        this->blockLength = (totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum(); // 均分到每个block，向上取整
        this->tileNum = tileNum;
        // 每个block进一步切tile
        uint32_t denom = tileNum * BUFFER_NUM;
        this->tileLength = (this->blockLength + denom - 1) / denom; // 向上取整，最后一tile做尾部处理

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE));
    }

    __aicore__ inline void Process()
    {
        const uint32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * this->tileLength;
            if (offset >= this->blockLength) {
                break;
            }
            uint32_t curLen = AscendC::min(this->tileLength, this->blockLength - offset);
            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm[offset], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> yLocal = outQueueY.AllocTensor<DTYPE>();
        // 基于文档：当src<0时，dst=src*scalar；否则dst=src
        AscendC::LeakyRelu(yLocal, xLocal, this->negSlope, static_cast<int32_t>(len));
        outQueueY.EnQue<DTYPE>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE> yLocal = outQueueY.DeQue<DTYPE>();
        AscendC::DataCopy(yGm[offset], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE> xGm;
    AscendC::GlobalTensor<DTYPE> yGm;

    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
    float negSlope = 0.01f;
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum, tilingData.negativeSlope);
    op.Process();
}
