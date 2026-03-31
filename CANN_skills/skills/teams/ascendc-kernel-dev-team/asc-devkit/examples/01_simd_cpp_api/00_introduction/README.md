# Introduce样例介绍

## 概述

基于Ascend C的简单的示例，通过Ascend C编程语言实现了自定义算子，分别给出对应的<<<>>>直调实现，适合初学者

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_quickstart](./00_quickstart) | 样例介绍了基于Ascend C的HelloWorld算子的核函数直调方法，从NPU侧运行核验证算子核函数，展示核函数从调用到执行的整体流程 |
| [01_add](./01_add) | 本样例介绍了基于Ascend C的Add自定义Vector算子的核函数直调方法，实现两个输入张量的逐元素相加，支持main函数和kernel函数在同一个cpp文件中实现 |
| [02_matrix](./02_matrix) | 本样例介绍了基于Ascend C的Matmul算子的核函数直调方法，可最大化利用AI处理器的并行计算能力，显著提升算子的执行效率，使用与高性能推理与训练场景 |
| [03_matrix_vector_fusion](./03_matrix_vector_fusion) | 本样例介绍了基于Ascend C的Matmul与Vector自定义融合算子的核函数直调方法，能够完成矩阵乘加与LeakyReLU激活的融合计算，以及SIMT Gather与SIMD Adds的融合计算，该方式将关键计算步骤在硬件层面高效协同执行，显著降低内存访问开销与计算延时 |
| [04_reg_base_add_compute](./04_reg_base_add_compute) | 本样例介绍了基于Ascend C的Add算子核函数直调方法（RegBase场景），实现两个输入张量的逐元素相加，展示了片上存储和寄存器层级的向量计算流程。|
