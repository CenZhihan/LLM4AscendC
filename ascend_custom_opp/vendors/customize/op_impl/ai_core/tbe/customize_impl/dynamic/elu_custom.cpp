
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelElu {
public:
    __aicore__ inline KernelElu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, float alpha)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->alpha = alpha;

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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // yLocal = abs(x)
        AscendC::Abs(yLocal, xLocal, this->tileLength);
        // yLocal = (abs(x) + x)
        AscendC::Add(yLocal, yLocal, xLocal, this->tileLength);
        // yLocal = 0.5 * (abs(x) + x) = pos
        AscendC::Muls(yLocal, yLocal, static_cast<DTYPE_Y>(0.5), this->tileLength);

        // xLocal = x - pos = neg
        AscendC::Sub(xLocal, xLocal, yLocal, this->tileLength);

        // xLocal = exp(neg)
        AscendC::Exp(xLocal, xLocal, this->tileLength);
        // xLocal = exp(neg) - 1
        AscendC::Adds(xLocal, xLocal, static_cast<DTYPE_X>(-1.0), this->tileLength);
        // xLocal = alpha * (exp(neg) - 1)
        AscendC::Muls(xLocal, xLocal, static_cast<DTYPE_X>(alpha), this->tileLength);

        // yLocal = pos + alpha*(exp(neg)-1)
        AscendC::Add(yLocal, yLocal, xLocal, this->tileLength);

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
    float alpha;
};

extern "C" __global__ __aicore__ void elu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelElu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.alpha);
    op.Process();
}
