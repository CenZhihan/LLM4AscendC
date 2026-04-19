# BestPractices样例介绍
## 概述
基于Ascend C的性能优化实践，聚焦于关键算子与内存访问的调优，旨在提升在Ascend平台上的运行效率。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_vector_compute_practices](./00_vector_compute_practices) | 基于Vector Compute API使用的优化样例，通过<<<>>>直调的实现方式，介绍基于静态Tensor方式编程的性能调优方法。 |
| [02_reg_vector_compute_practices](./02_reg_vector_compute_practices) | 基于VF函数的性能优化样例，通过<<<>>>直调的实现方式，介绍了VF循环优化、VF指令双发优化、VF连续非对齐场景优化、VF融合优化的方法。 |
| [03_fusion_compute_practices](./03_fusion_compute_practices) | 基于SIMT与SIMD混合编程方式实现的算子样例，介绍基于SIMT灵活实现分支判断逻辑，以及Matmul融合算子的高性能实现和使用UB提升离散内存访问效率的性能优化方式。 |
| [04_memory_access_practices](./04_memory_access_practices) | 基于搬运类API使用的优化样例，通过<<<>>>直调的实现方式，介绍了减少无效数据搬运、减少搬运指令数量等方法。 |