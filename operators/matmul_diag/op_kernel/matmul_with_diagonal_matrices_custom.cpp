
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void matmul_with_diagonal_matrices_custom(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    uint32_t totalRows = tilingData.rows;
    uint32_t cols = tilingData.cols;
    uint32_t tileLength = tilingData.tileLength;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockDim = AscendC::GetBlockNum();
    uint32_t rowsPerBlock = totalRows / blockDim;
    uint32_t remain = totalRows % blockDim;

    uint32_t startRow = (blockIdx < remain) ? blockIdx * (rowsPerBlock + 1)
                                            : remain * (rowsPerBlock + 1) + (blockIdx - remain) * rowsPerBlock;
    uint32_t endRow = (blockIdx < remain) ? startRow + rowsPerBlock + 1 : startRow + rowsPerBlock;
    uint32_t currentRows = endRow - startRow;
    if (currentRows == 0) return;

    constexpr uint32_t BUFFER_NUM = 2;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    pipe.InitBuffer(inQueueA, 1, (currentRows + 32) * sizeof(float));
    pipe.InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(float));
    pipe.InitBuffer(outQueueC, BUFFER_NUM, tileLength * sizeof(float));

    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A), totalRows);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B), totalRows * cols);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(C), totalRows * cols);

    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
    AscendC::DataCopy(aLocal, aGm[startRow], currentRows);
    inQueueA.EnQue(aLocal);
    aLocal = inQueueA.DeQue<float>();

    for (uint32_t i = 0; i < currentRows; ++i) {
        float scalarA = aLocal.GetValue(i);
        uint32_t rowOffset = (startRow + i) * cols;
        for (uint32_t j = 0; j < cols; j += tileLength) {
            uint32_t curTile = (j + tileLength > cols) ? (cols - j) : tileLength;
            AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
            AscendC::DataCopy(bLocal, bGm[rowOffset + j], curTile);
            inQueueB.EnQue(bLocal);

            bLocal = inQueueB.DeQue<float>();
            AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
            AscendC::Muls(cLocal, bLocal, scalarA, curTile);
            outQueueC.EnQue(cLocal);
            inQueueB.FreeTensor(bLocal);

            cLocal = outQueueC.DeQue<float>();
            AscendC::DataCopy(cGm[rowOffset + j], cLocal, curTile);
            outQueueC.FreeTensor(cLocal);
        }
    }
    inQueueA.FreeTensor(aLocal);
}

