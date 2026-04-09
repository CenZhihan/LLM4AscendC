
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3dScalingTanhMultiplySigmoid {
public:
    __aicore__ inline KernelConv3dScalingTanhMultiplySigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scaling, GM_ADDR bias, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sGm.SetGlobalBuffer((__gm__ float *)scaling + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ float *)bias + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueS, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.AllocTensor<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(sLocal, sGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueS.EnQue(sLocal);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // yLocal = x * scaling
        AscendC::Mul(yLocal, xLocal, sLocal, this->tileLength);
        // yLocal = tanh(yLocal)
        AscendC::Tanh(yLocal, yLocal, this->tileLength);
        // yLocal = yLocal * bias
        AscendC::Mul(yLocal, yLocal, bLocal, this->tileLength);
        // yLocal = sigmoid(yLocal)
        AscendC::Sigmoid(yLocal, yLocal, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueS.FreeTensor(sLocal);
        inQueueB.FreeTensor(bLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueS, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> sGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv3d_scaling_tanh_multiply_sigmoid_custom(
    GM_ADDR x, GM_ADDR scaling, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dScalingTanhMultiplySigmoid op;
    op.Init(x, scaling, bias, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
