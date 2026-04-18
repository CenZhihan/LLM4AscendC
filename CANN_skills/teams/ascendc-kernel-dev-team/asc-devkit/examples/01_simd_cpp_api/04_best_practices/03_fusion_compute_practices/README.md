# Fusion Compute Practices样例介绍
## 概述
基于SIMT与SIMD混合编程方式实现的算子样例，介绍基于SIMT灵活实现分支判断逻辑，以及Matmul融合算子的高性能实现和使用UB提升离散内存访问效率的性能优化方式。
 
## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [grouped_matmul](./grouped_matmul) | 本样例介绍QuantGroupMatmul算子在NPU上高性能实现，支持分组量化矩阵乘与Gelu激活计算。 |
| [simt_and_simd_floor_mod](./simt_and_simd_floor_mod) | 基于SIMT与SIMD混合编程方式实现的算子样例，介绍基于SIMT灵活实现分支判断逻辑。 |
| [simt_gather_with_ub](./simt_gather_with_ub) | 本样例以Gather算子为示例，展示了在SIMD与SIMT混合编程模式下模式下使用UB提升离散内存访问效率的性能优化方式 |