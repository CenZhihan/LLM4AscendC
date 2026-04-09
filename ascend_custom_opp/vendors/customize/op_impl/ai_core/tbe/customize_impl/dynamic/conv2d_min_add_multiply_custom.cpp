
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dMinAddMultiply {
public:
    __aicore__ inline KernelConv2dMinAddMultiply() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR c, GM_ADDR s, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / (tileNum * BUFFER_NUM);

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        biasGm.SetGlobalBuffer((__gm__ float *)bias + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        cGm.SetGlobalBuffer((__gm__ float *)c + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sGm.SetGlobalBuffer((__gm__ float *)s + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueS, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> bLocal = inQueueBias.AllocTensor<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, biasGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(cLocal, cGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(sLocal, sGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueBias.EnQue(bLocal);
        inQueueC.EnQue(cLocal);
        inQueueS.EnQue(sLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.DeQue<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // z = min(x, c) + bias
        AscendC::Min(zLocal, xLocal, cLocal, this->tileLength);
        AscendC::Add(zLocal, zLocal, bLocal, this->tileLength);
        // z = z * s
        AscendC::Mul(zLocal, zLocal, sLocal, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueBias.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
        inQueueS.FreeTensor(sLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueBias, inQueueC, inQueueS;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::GlobalTensor<float> sGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_min_add_multiply_custom(GM_ADDR x,
                                                                      GM_ADDR bias,
                                                                      GM_ADDR c,
                                                                      GM_ADDR s,
                                                                      GM_ADDR z,
                                                                      GM_ADDR workspace,
                                                                      GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMinAddMultiply op;
    op.Init(x, bias, c, s, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
