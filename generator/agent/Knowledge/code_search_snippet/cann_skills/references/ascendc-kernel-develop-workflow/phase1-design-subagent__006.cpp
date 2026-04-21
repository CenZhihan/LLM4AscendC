// 1. Tiling 结构体（独立定义）
struct {operator_name}{分支}Tiling {
    uint32_t param1;
    uint32_t param2;
};

// 2. 判断函数（判断是否应走此分支）
__inline__ bool Is{operator_name}{分支}Case(参数列表) {
    return 判断条件;
}

// 3. Kernel 实现
class Kernel{operator_name}{分支} {
public:
    __aicore__ inline void Process(...);
};
