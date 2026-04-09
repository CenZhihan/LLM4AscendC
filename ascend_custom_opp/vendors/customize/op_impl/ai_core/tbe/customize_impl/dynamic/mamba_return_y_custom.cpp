
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelMambaReturnYCustom {
public:
    __aicore__ inline KernelMambaReturnYCustom() {}

    __aicore__ inline void Init(GM_ADDR y_in, GM_ADDR y_out, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // split by double buffering
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        inGm.SetGlobalBuffer((__gm__ float *)y_in + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float *)y_out + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(inLocal, inGm[progress * this->tileLength], this->tileLength);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t)
    {
        AscendC::LocalTensor<float> inLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        // Identity copy within local buffers
        AscendC::DataCopy(outLocal, inLocal, this->tileLength);
        outQueue.EnQue<float>(outLocal);
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> inGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void mamba_return_y_custom(GM_ADDR y_in, GM_ADDR y_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMambaReturnYCustom op;
    op.Init(y_in, y_out, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
