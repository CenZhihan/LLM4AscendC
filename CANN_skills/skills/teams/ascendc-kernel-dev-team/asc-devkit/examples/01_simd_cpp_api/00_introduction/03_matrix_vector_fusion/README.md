# MatmulLeakyRelu 样例介绍

## 概述

本样例介绍了基于Ascend C的Matmul与Vector自定义融合算子的核函数直调方法，能够完成矩阵乘加与LeakyReLU激活的融合计算，以及SIMT Gather与SIMD Adds的融合计算，该方式将关键计算步骤在硬件层面高效协同执行，显著降低内存访问开销与计算延时。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [matmul_leakyrelu](./matmul_leakyrelu) | 本样例基于Ascend与其他演示MatmulLeakyRelu算子的核函数直调实现。通过将矩阵乘加（Matmul）与LeakyRelu激活函数计算融合，降低内存访问开销 |
| [simt_gather_and_simd_adds](./simt_gather_and_simd_adds) | 本样例基于Ascend C演示SimtGatherAndSimdAdds算子的核函数直调实现。通过将SIMT Gather与SIMD Adds计算融合，降低内存访问开销 |