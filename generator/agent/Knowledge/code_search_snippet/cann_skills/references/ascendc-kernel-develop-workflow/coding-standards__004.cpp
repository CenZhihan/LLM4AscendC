// 1. Kernel 类定义
class Kernel{Operator} {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECOUT, 1> outQueueY;
};

// 2. Kernel 入口函数（必须在调用之前定义）
__global__ __aicore__ void {operator_name}_custom(GM_ADDR x, GM_ADDR y) {
    Kernel{Operator} op;
    op.Process(x, y);
}

// 3. Host 侧调用
extern "C" void {operator_name}_custom_do(...) {
    Kernel{Operator}<<<blockDim, l2ctrl>>>();
}
