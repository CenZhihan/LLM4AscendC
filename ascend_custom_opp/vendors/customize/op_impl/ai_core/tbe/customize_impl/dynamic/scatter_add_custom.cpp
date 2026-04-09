
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelScatterAdd {
public:
    __aicore__ inline KernelScatterAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, GM_ADDR updates, GM_ADDR y, uint32_t N, uint32_t C, uint32_t K)
    {
        this->N = N;
        this->C = C;
        this->K = K;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint32_t)(N * C));
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint32_t)(N * C));
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, (uint32_t)(N * K));
        updGm.SetGlobalBuffer((__gm__ float*)updates, (uint32_t)(N * K));

        pipe.InitBuffer(inQueueIdx, BUFFER_NUM, K * sizeof(int32_t));
        pipe.InitBuffer(inQueueUpd, BUFFER_NUM, K * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, C * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowsPerBlock = (N + blockNum - 1) / blockNum;
        uint32_t rowStart = blockIdx * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > N) rowEnd = N;

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            CopyIn(r);
            Compute(r);
            CopyOut(r);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t r)
    {
        LocalTensor<int32_t> idxLocal = inQueueIdx.AllocTensor<int32_t>();
        DataCopy(idxLocal, idxGm[r * K], K);
        inQueueIdx.EnQue(idxLocal);

        LocalTensor<float> updLocal = inQueueUpd.AllocTensor<float>();
        DataCopy(updLocal, updGm[r * K], K);
        inQueueUpd.EnQue(updLocal);
    }

    __aicore__ inline void Compute(uint32_t r)
    {
        LocalTensor<int32_t> idxLocal = inQueueIdx.DeQue<int32_t>();
        LocalTensor<float> updLocal = inQueueUpd.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Initialize yLocal with x row
        DataCopy(yLocal, xGm[r * C], C);

        __ubuf__ float* yAddr = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
        __ubuf__ const float* updAddr = reinterpret_cast<__ubuf__ const float*>(updLocal.GetPhyAddr());
        __ubuf__ const int32_t* idxAddr = reinterpret_cast<__ubuf__ const int32_t*>(idxLocal.GetPhyAddr());

        for (uint32_t i = 0; i < K; ++i) {
            int32_t col = idxAddr[i];
            // assume 0 <= col < C
            yAddr[col] += updAddr[i];
        }

        outQueueY.EnQue(yLocal);
        inQueueIdx.FreeTensor(idxLocal);
        inQueueUpd.FreeTensor(updLocal);
    }

    __aicore__ inline void CopyOut(uint32_t r)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[r * C], yLocal, C);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueIdx;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueUpd;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<float> updGm;

    uint32_t N;
    uint32_t C;
    uint32_t K;
};

extern "C" __global__ __aicore__ void scatter_add_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelScatterAdd op;
    op.Init(x, idx, updates, y, tiling_data.N, tiling_data.C, tiling_data.K);
    op.Process();
}
