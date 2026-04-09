
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM_COPY = 2;

class KernelInplaceUpdate {
public:
    __aicore__ inline KernelInplaceUpdate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, GM_ADDR value, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum,
                                uint32_t rows, uint32_t cols, uint32_t idxLen)
    {
        this->rows = rows;
        this->cols = cols;
        this->idxLen = idxLen;
        this->tileNum = tileNum;
        this->totalLength = totalLength;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Balanced partition for copy phase
        this->blkCopyStart = (totalLength * blockIdx) / blockNum;
        uint32_t nextStart = (totalLength * (blockIdx + 1)) / blockNum;
        this->blkCopyLen = nextStart - this->blkCopyStart;

        // Index/value partition for update phase
        this->idxStart = (idxLen * blockIdx) / blockNum;
        uint32_t idxNext = (idxLen * (blockIdx + 1)) / blockNum;
        this->idxCount = idxNext - this->idxStart;

        // Set global buffers
        xGm.SetGlobalBuffer((__gm__ float*)x + this->blkCopyStart, this->blkCopyLen);
        zPartGm.SetGlobalBuffer((__gm__ float*)z + this->blkCopyStart, this->blkCopyLen);
        zAllGm.SetGlobalBuffer((__gm__ float*)z, totalLength);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx + this->idxStart, this->idxCount);
        valueGm.SetGlobalBuffer((__gm__ float*)value + (uint64_t)this->idxStart * cols, (uint64_t)this->idxCount * cols);

        // Initialize buffers
        this->tileLenCopy = this->blkCopyLen / (tileNum * BUFFER_NUM_COPY);
        if (this->tileLenCopy == 0) { this->tileLenCopy = this->blkCopyLen > 0 ? this->blkCopyLen : 1; }

        pipe.InitBuffer(inQueueCopy, BUFFER_NUM_COPY, this->tileLenCopy * sizeof(float));
        pipe.InitBuffer(idxQueue, 1, sizeof(int32_t));
        pipe.InitBuffer(valQueue, 1, this->cols * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Phase 1: copy original x to z
        CopyOriginalToOutput();
        // Phase 2: apply indexed row updates
        ApplyUpdates();
    }

private:
    __aicore__ inline void CopyOriginalToOutput()
    {
        uint32_t processed = 0;
        while (processed < this->blkCopyLen) {
            uint32_t thisLen = this->tileLenCopy;
            if (thisLen > (this->blkCopyLen - processed)) {
                thisLen = this->blkCopyLen - processed;
            }
            AscendC::LocalTensor<float> buf = inQueueCopy.AllocTensor<float>();
            AscendC::DataCopy(buf, xGm[processed], thisLen);
            inQueueCopy.EnQue(buf);

            AscendC::LocalTensor<float> bufDeq = inQueueCopy.DeQue<float>();
            AscendC::DataCopy(zPartGm[processed], bufDeq, thisLen);
            inQueueCopy.FreeTensor(bufDeq);

            processed += thisLen;
        }
    }

    __aicore__ inline void ApplyUpdates()
    {
        for (uint32_t r = 0; r < this->idxCount; ++r) {
            // Load index
            AscendC::LocalTensor<int32_t> idxLocal = idxQueue.AllocTensor<int32_t>();
            AscendC::DataCopy(idxLocal, idxGm[r], 1);
            idxQueue.EnQue(idxLocal);
            AscendC::LocalTensor<int32_t> idxLocalDeq = idxQueue.DeQue<int32_t>();
            int32_t rowIndex = idxLocalDeq.GetValue(0);
            idxQueue.FreeTensor(idxLocalDeq);

            if (rowIndex < 0 || (uint32_t)rowIndex >= this->rows) {
                continue; // out of range protection
            }

            // Load one row of value
            AscendC::LocalTensor<float> valLocal = valQueue.AllocTensor<float>();
            AscendC::DataCopy(valLocal, valueGm[(uint64_t)r * this->cols], this->cols);
            valQueue.EnQue(valLocal);
            AscendC::LocalTensor<float> valLocalDeq = valQueue.DeQue<float>();

            // Store into z at the target row
            uint64_t dstOffset = (uint64_t)rowIndex * this->cols;
            AscendC::DataCopy(zAllGm[dstOffset], valLocalDeq, this->cols);

            valQueue.FreeTensor(valLocalDeq);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM_COPY> inQueueCopy;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idxQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> valQueue;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zPartGm;
    AscendC::GlobalTensor<float> zAllGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> valueGm;

    uint32_t rows;
    uint32_t cols;
    uint32_t idxLen;

    uint32_t totalLength;
    uint32_t blkCopyStart;
    uint32_t blkCopyLen;

    uint32_t idxStart;
    uint32_t idxCount;

    uint32_t tileNum;
    uint32_t tileLenCopy;
};

extern "C" __global__ __aicore__ void inplace_update_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR value, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInplaceUpdate op;
    op.Init(x, idx, value, z,
            tiling_data.totalLength,
            tiling_data.tileNum,
            tiling_data.rows,
            tiling_data.cols,
            tiling_data.idxLen);
    op.Process();
}
