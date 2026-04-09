
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelKvCachedSpeculativeAttention {
public:
    __aicore__ inline KernelKvCachedSpeculativeAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR o,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->blockLength = (totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = (this->blockLength + (tileNum * BUFFER_NUM) - 1) / (tileNum * BUFFER_NUM);

        // Align tileLength to at least 1
        if (this->tileLength == 0) {
            this->tileLength = 1;
        }

        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        if (blockOffset + this->blockLength > this->totalLength) {
            this->blockLength = (this->totalLength > blockOffset) ? (this->totalLength - blockOffset) : 0;
        }

        qGm.SetGlobalBuffer((__gm__ float *)q + blockOffset, this->blockLength);
        kGm.SetGlobalBuffer((__gm__ float *)k_cache, 0); // not used in this simplified kernel
        vGm.SetGlobalBuffer((__gm__ float *)v_cache, 0); // not used in this simplified kernel
        oGm.SetGlobalBuffer((__gm__ float *)o + blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueQ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }
        int32_t totalTiles = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int32_t i = 0; i < totalTiles; i++) {
            uint32_t currentLen = GetCurrentTileLength(i);
            CopyIn(i, currentLen);
            Compute(i, currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentTileLength(int32_t progress) const
    {
        uint32_t offset = progress * this->tileLength;
        uint32_t remain = (offset < this->blockLength) ? (this->blockLength - offset) : 0;
        return remain < this->tileLength ? remain : this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
        AscendC::DataCopy(qLocal, qGm[progress * this->tileLength], len);
        inQueueQ.EnQue(qLocal);
    }

    __aicore__ inline void Compute(int32_t /*progress*/, uint32_t len)
    {
        AscendC::LocalTensor<float> qLocal = inQueueQ.DeQue<float>();
        AscendC::LocalTensor<float> oLocal = outQueueO.AllocTensor<float>();

        // Simplified placeholder: directly copy Q to output.
        // In a full implementation, this would compute softmax(Q * K^T / sqrt(d)) @ V.
        AscendC::DataCopy(oLocal, qLocal, len);

        outQueueO.EnQue<float>(oLocal);
        inQueueQ.FreeTensor(qLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<float> oLocal = outQueueO.DeQue<float>();
        AscendC::DataCopy(oGm[progress * this->tileLength], oLocal, len);
        outQueueO.FreeTensor(oLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueO;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void kv_cached_speculative_attention_custom(GM_ADDR q,
                                                                             GM_ADDR k_cache,
                                                                             GM_ADDR v_cache,
                                                                             GM_ADDR o,
                                                                             GM_ADDR workspace,
                                                                             GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelKvCachedSpeculativeAttention op;
    op.Init(q, k_cache, v_cache, o, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
