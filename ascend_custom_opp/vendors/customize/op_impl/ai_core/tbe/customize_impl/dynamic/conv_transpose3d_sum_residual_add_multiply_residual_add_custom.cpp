
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelFused {
public:
    __aicore__ inline KernelFused() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR res, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ float *)bias + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        rGm.SetGlobalBuffer((__gm__ float *)res + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(rLocal, rGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueB.EnQue(bLocal);
        inQueueR.EnQue(rLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> rLocal = inQueueR.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // z = (((x + bias) + res) * res) + res
        AscendC::Add(zLocal, xLocal, bLocal, this->tileLength);      // z = x + bias
        AscendC::Add(zLocal, zLocal, rLocal, this->tileLength);      // z = z + res
        AscendC::Mul(zLocal, zLocal, rLocal, this->tileLength);      // z = z * res
        AscendC::Add(zLocal, zLocal, rLocal, this->tileLength);      // z = z + res

        outQueueZ.EnQue<float>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueR.FreeTensor(rLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueB, inQueueR;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> rGm;
    AscendC::GlobalTensor<float> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_sum_residual_add_multiply_residual_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR res, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, res, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
