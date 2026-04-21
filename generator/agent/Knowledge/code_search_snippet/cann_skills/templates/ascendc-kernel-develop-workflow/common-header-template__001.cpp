#ifndef {OPERATOR}_COMMON_H
#define {OPERATOR}_COMMON_H

#include "kernel_operator.h"

// Tiling 结构体
struct {Operator}TilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    // ... 其他参数
};

// 分支枚举（如有多分支）
enum class BranchType {
    BRANCH_1,
    BRANCH_2,
    // ...
};

// Host 侧分支判断函数
inline BranchType DetermineBranch(const {Operator}TilingData& tiling) {
    // ...
}

// Host 侧 Tiling 计算函数
inline void ComputeTiling({Operator}TilingData& tiling, /* 参数 */) {
    // ...
}

#endif // {OPERATOR}_COMMON_H
