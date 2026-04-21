# 12_high_performance_vf样例

## 概述

基于VF函数的性能优化样例，通过<<<>>>直调的实现方式，介绍了VF循环优化、VF指令双发优化、VF连续非对齐场景优化、VF融合优化的方法。

## 算子开发样例
|  目录名称  |  功能描述  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [optimize_vf_continious_align](./optimize_vf_continious_align) | 本样例演示了SIMD场景下，基于连续非对齐搬运接口LoadUnAlign/StoreUnAlign进行搬运优化的算子实现。 |
| [optimize_vf_dual_instr](./optimize_vf_dual_instr) | 本样例演示了SIMD场景下，基于RegBase编程范式下VF指令双发优化的样例，通过合理拆分VF循环，适当搬出中间结果到UB，减少数据依赖。 |
| [optimize_vf_fusion](./optimize_vf_fusion) | 本样例演示了SIMD场景下，基于RegBase编程范式，通过VF融合优化算子代码实现。 |
| [optimize_vf_loop](./optimize_vf_loop) | 通过循环内成员变量访问优化、循环内指令分布优化、循环内地址管理优化等手段优化VF循环。 |
