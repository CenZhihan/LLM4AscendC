
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAdagrad {
public:
    __aicore__ inline KernelAdagrad() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR accum, GM_ADDR lr, GM_ADDR eps, GM_ADDR out,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        paramGm.SetGlobalBuffer((__gm__ DTYPE_PARAM *)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ DTYPE_GRAD *)grad + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        accumGm.SetGlobalBuffer((__gm__ DTYPE_ACCUM *)accum + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        lrGm.SetGlobalBuffer((__gm__ DTYPE_LR *)lr + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        epsGm.SetGlobalBuffer((__gm__ DTYPE_EPS *)eps + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT *)out + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(DTYPE_PARAM));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(DTYPE_GRAD));
        pipe.InitBuffer(inQueueAccum, BUFFER_NUM, this->tileLength * sizeof(DTYPE_ACCUM));
        pipe.InitBuffer(inQueueLr, BUFFER_NUM, this->tileLength * sizeof(DTYPE_LR));
        pipe.InitBuffer(inQueueEps, BUFFER_NUM, this->tileLength * sizeof(DTYPE_EPS));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(DTYPE_OUT));
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
        AscendC::LocalTensor<DTYPE_PARAM> paramLocal = inQueueParam.AllocTensor<DTYPE_PARAM>();
        AscendC::LocalTensor<DTYPE_GRAD> gradLocal = inQueueGrad.AllocTensor<DTYPE_GRAD>();
        AscendC::LocalTensor<DTYPE_ACCUM> accumLocal = inQueueAccum.AllocTensor<DTYPE_ACCUM>();
        AscendC::LocalTensor<DTYPE_LR> lrLocal = inQueueLr.AllocTensor<DTYPE_LR>();
        AscendC::LocalTensor<DTYPE_EPS> epsLocal = inQueueEps.AllocTensor<DTYPE_EPS>();

        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(accumLocal, accumGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(lrLocal, lrGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(epsLocal, epsGm[progress * this->tileLength], this->tileLength);

        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueAccum.EnQue(accumLocal);
        inQueueLr.EnQue(lrLocal);
        inQueueEps.EnQue(epsLocal);
    }
    __aicore__ inline void Compute(int32_t)
    {
        AscendC::LocalTensor<DTYPE_PARAM> paramLocal = inQueueParam.DeQue<DTYPE_PARAM>();
        AscendC::LocalTensor<DTYPE_GRAD> gradLocal = inQueueGrad.DeQue<DTYPE_GRAD>();
        AscendC::LocalTensor<DTYPE_ACCUM> accumLocal = inQueueAccum.DeQue<DTYPE_ACCUM>();
        AscendC::LocalTensor<DTYPE_LR> lrLocal = inQueueLr.DeQue<DTYPE_LR>();
        AscendC::LocalTensor<DTYPE_EPS> epsLocal = inQueueEps.DeQue<DTYPE_EPS>();
        AscendC::LocalTensor<DTYPE_OUT> outLocal = outQueueOut.AllocTensor<DTYPE_OUT>();

        // temp: outLocal = grad * grad
        AscendC::Mul(outLocal, gradLocal, gradLocal, this->tileLength);
        // accumLocal = accumLocal + outLocal (new accum not written back)
        AscendC::Add(accumLocal, accumLocal, outLocal, this->tileLength);
        // outLocal = sqrt(accumLocal)
        AscendC::Sqrt(outLocal, accumLocal, this->tileLength);
        // outLocal = outLocal + epsLocal
        AscendC::Add(outLocal, outLocal, epsLocal, this->tileLength);
        // outLocal = gradLocal / outLocal
        AscendC::Div(outLocal, gradLocal, outLocal, this->tileLength);
        // outLocal = outLocal * lrLocal
        AscendC::Mul(outLocal, outLocal, lrLocal, this->tileLength);
        // outLocal = paramLocal - outLocal
        AscendC::Sub(outLocal, paramLocal, outLocal, this->tileLength);

        outQueueOut.EnQue<DTYPE_OUT>(outLocal);

        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueAccum.FreeTensor(accumLocal);
        inQueueLr.FreeTensor(lrLocal);
        inQueueEps.FreeTensor(epsLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_OUT> outLocal = outQueueOut.DeQue<DTYPE_OUT>();
        AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueGrad, inQueueAccum, inQueueLr, inQueueEps;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOut;

    AscendC::GlobalTensor<DTYPE_PARAM> paramGm;
    AscendC::GlobalTensor<DTYPE_GRAD> gradGm;
    AscendC::GlobalTensor<DTYPE_ACCUM> accumGm;
    AscendC::GlobalTensor<DTYPE_LR> lrGm;
    AscendC::GlobalTensor<DTYPE_EPS> epsGm;
    AscendC::GlobalTensor<DTYPE_OUT> outGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void adagrad_custom(GM_ADDR param, GM_ADDR grad, GM_ADDR accum, GM_ADDR lr, GM_ADDR eps, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdagrad op;
    op.Init(param, grad, accum, lr, eps, out, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
