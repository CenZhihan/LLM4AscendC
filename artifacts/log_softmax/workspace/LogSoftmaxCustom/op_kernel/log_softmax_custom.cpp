
#include "kernel_operator.h"
#define __NPU_TILING__
#include "log_softmax_custom_tiling_data.h"

class KernelLogSoftmax {
public:
    __aicore__ inline KernelLogSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLen)
    {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalLen);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, totalLen);
        totalLength = totalLen;
    }

    __aicore__ inline void Process(const LogSoftMaxTiling &tilingParam)
    {
        // 直接调用高阶API，使用Tiling参数完成整张量的LogSoftMax计算
        // 假设LogSoftMax默认沿最后一维（与Host传入的shape一致）
        AscendC::LogSoftMax<DTYPE_X, false>(yGm, xGm, tilingParam);
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t totalLength = 0;
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSoftmax op;
    op.Init(x, y, tilingData.totalLength);
    op.Process(tilingData.logSoftmaxTilingData);
}
