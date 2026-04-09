
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

// Implements GELU approximation used in GPT/BERT:
// y = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
class KernelMiniGptBlock {
public:
    __aicore__ inline KernelMiniGptBlock() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        // Use single output local tensor for intermediate computations to reduce buffer use.
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // yLocal will be used as temp and final output buffer.

        // Step 1: yLocal = x^2
        AscendC::Mul(yLocal, xLocal, xLocal, this->tileLength);

        // Step 2: yLocal = x^3
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);

        // Step 3: yLocal = 0.044715 * x^3
        const float kCubicCoef = 0.044715f;
        AscendC::Muls(yLocal, yLocal, kCubicCoef, this->tileLength);

        // Step 4: yLocal = x + 0.044715 * x^3
        AscendC::Add(yLocal, yLocal, xLocal, this->tileLength);

        // Step 5: yLocal = sqrt(2/pi) * (x + ...)
        const float kSqrt2OverPi = 0.7978845834732056f; // sqrt(2/pi)
        AscendC::Muls(yLocal, yLocal, kSqrt2OverPi, this->tileLength);

        // Step 6: yLocal = tanh(yLocal)
        AscendC::Tanh(yLocal, yLocal, this->tileLength);

        // Step 7: yLocal = yLocal + 1
        AscendC::Adds(yLocal, yLocal, 1.0f, this->tileLength);

        // Step 8: xLocal = 0.5 * x
        AscendC::Muls(xLocal, xLocal, 0.5f, this->tileLength);

        // Step 9: yLocal = (1 + tanh(...)) * (0.5 * x)
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void mini_gpt_block_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMiniGptBlock op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
