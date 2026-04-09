
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

// A simplified kernel: copies V to O as a placeholder for custom attention.
// The interface and tiling follow the custom op so it can be replaced by a more advanced version later.
class KernelScaledDotProductAttentionCustom {
public:
    __aicore__ inline KernelScaledDotProductAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // We tile across the flattened length [B*H*S*D].
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        qGm.SetGlobalBuffer((__gm__ float *)q + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        kGm.SetGlobalBuffer((__gm__ float *)k + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vGm.SetGlobalBuffer((__gm__ float *)v + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        oGm.SetGlobalBuffer((__gm__ float *)o + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(vInQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(oOutQueue, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> vLocal = vInQueue.AllocTensor<float>();
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        vInQueue.EnQue(vLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        // Placeholder compute: O = V
        AscendC::LocalTensor<float> vLocal = vInQueue.DeQue<float>();
        AscendC::LocalTensor<float> oLocal = oOutQueue.AllocTensor<float>();
        AscendC::DataCopy(oLocal, vLocal, this->tileLength);
        oOutQueue.EnQue<float>(oLocal);
        vInQueue.FreeTensor(vLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> oLocal = oOutQueue.DeQue<float>();
        AscendC::DataCopy(oGm[progress * this->tileLength], oLocal, this->tileLength);
        oOutQueue.FreeTensor(oLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> vInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> oOutQueue;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaledDotProductAttentionCustom op;
    op.Init(q, k, v, o, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
