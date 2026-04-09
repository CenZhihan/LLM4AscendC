
#include "kernel_operator.h"

class KernelVanillaRnn {
public:
    __aicore__ inline KernelVanillaRnn() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                uint32_t batchSize, uint32_t inputSize, uint32_t hiddenSize) {
        xPtr = reinterpret_cast<__gm__ float*>(x);
        hPtr = reinterpret_cast<__gm__ float*>(h);
        wPtr = reinterpret_cast<__gm__ float*>(w);
        bPtr = reinterpret_cast<__gm__ float*>(b);
        yPtr = reinterpret_cast<__gm__ float*>(y);
        B = batchSize;
        I = inputSize;
        H = hiddenSize;

        blockNum = AscendC::GetBlockNum();
        blockId = AscendC::GetBlockIdx();

        // simple row-wise split along batch
        rowsPerBlock = (B + blockNum - 1) / blockNum;
        rowStart = blockId * rowsPerBlock;
        rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > B) rowEnd = B;
    }

    __aicore__ inline void Process() {
        // Compute y[b, j] = sum_{k=0..I-1} x[b,k] * w[j, k] + sum_{k=0..H-1} h[b,k] * w[j, I+k] + b[j]
        for (uint32_t b = rowStart; b < rowEnd; ++b) {
            __gm__ float* xRow = xPtr + static_cast<uint64_t>(b) * I;
            __gm__ float* hRow = hPtr + static_cast<uint64_t>(b) * H;
            __gm__ float* yRow = yPtr + static_cast<uint64_t>(b) * H;

            for (uint32_t j = 0; j < H; ++j) {
                // weight row pointer: w[j, :]
                __gm__ float* wRow = wPtr + static_cast<uint64_t>(j) * (I + H);
                float acc = bPtr[j];

                // accumulate x part
                for (uint32_t k = 0; k < I; ++k) {
                    acc += xRow[k] * wRow[k];
                }
                // accumulate h part
                for (uint32_t k = 0; k < H; ++k) {
                    acc += hRow[k] * wRow[I + k];
                }
                yRow[j] = acc;
            }
        }
    }

private:
    __gm__ float* xPtr;
    __gm__ float* hPtr;
    __gm__ float* wPtr;
    __gm__ float* bPtr;
    __gm__ float* yPtr;

    uint32_t B;
    uint32_t I;
    uint32_t H;

    uint32_t blockNum;
    uint32_t blockId;
    uint32_t rowsPerBlock;
    uint32_t rowStart;
    uint32_t rowEnd;
};

extern "C" __global__ __aicore__ void vanilla_rnn_custom(GM_ADDR x, GM_ADDR h, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVanillaRnn op;
    op.Init(x, h, weight, bias, y,
            tiling_data.batchSize, tiling_data.inputSize, tiling_data.hiddenSize);
    op.Process();
}
