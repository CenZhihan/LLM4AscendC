
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

// A simplified attention kernel that forwards V to output (placeholder for SDPA core).
class KernelVisionAttention {
public:
    __aicore__ inline KernelVisionAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        qGm.SetGlobalBuffer((__gm__ float *)q + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        kGm.SetGlobalBuffer((__gm__ float *)k + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vGm.SetGlobalBuffer((__gm__ float *)v + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        oGm.SetGlobalBuffer((__gm__ float *)o + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueQ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
        AscendC::LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        AscendC::DataCopy(qLocal, qGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(kLocal, kGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        inQueueQ.EnQue(qLocal);
        inQueueK.EnQue(kLocal);
        inQueueV.EnQue(vLocal);
    }
    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<float> qLocal = inQueueQ.DeQue<float>();
        AscendC::LocalTensor<float> kLocal = inQueueK.DeQue<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.DeQue<float>();
        AscendC::LocalTensor<float> oLocal = outQueueO.AllocTensor<float>();

        // Placeholder: o = v (bypass attention math)
        AscendC::DataCopy(oLocal, vLocal, this->tileLength);

        outQueueO.EnQue<float>(oLocal);
        inQueueQ.FreeTensor(qLocal);
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> oLocal = outQueueO.DeQue<float>();
        AscendC::DataCopy(oGm[progress * this->tileLength], oLocal, this->tileLength);
        outQueueO.FreeTensor(oLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueQ, inQueueK, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueO;
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vision_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVisionAttention op;
    op.Init(q, k, v, o, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
