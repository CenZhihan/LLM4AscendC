# 数据搬运类api样例介绍

## 概述

本路径下包含了与数据搬运相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

|  目录名称                                                  |  功能描述                                             |
| ----------------------------------------------------------- | --------------------------------------------------- |
| [copy](./copy) | 本样例基于Copy实现数据搬运，适用于VECIN，VECCALC，VECOUT之间的数据搬运，支持mask操作和DataBlock间隔操作 |
| [data_copy_pad](./data_copy_pad) | 本样例基于DataCopyPad实现数据非对齐搬运，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据 |
| [data_copy_ub2l1](./data_copy_ub2l1) | 本样例在Mmad矩阵乘场景下，基于DataCopy实现UB(Unified Buffer)到L1(L1 Buffer)的数据搬运 |
| [scalar_quantized_activation_during](./scalar_quantized_activation_during) | 本样例在卷积场景下，基于DataCopy实现数据随路量化激活搬运，支持在数据搬运过程中通过Scalar量化将int32_t类型转换为half类型 |
| [slice_data_movement](./slice_data_movement) | 本样例基于DataCopy实现数据切片搬运，提取多维Tensor数据的子集进行GM(Global Memory)与UB(Unified Buffer)通路之间的搬运 |
| [tensor_quantized_activation_during](./tensor_quantized_activation_during) | 本样例在卷积场景下，基于DataCopy实现数据随路量化激活搬运，支持在数据搬运过程中通过Tensor量化将int32_t类型转换为half类型 |
| [broadcast_vec_to_mm](./broad_cast_vec_to_mm) | 本样例基于BroadCastVecToMM实现数据广播搬运，适用于Unified Buffer与L0C Buffer之间的数据传输 |
| [multidimensional_data_movement](./multidimensional_data_movement) | 本样例介绍如何使用多维数据搬运接口实现GM(Global Memory)到UB(Unified Buffer)通路的数据搬运，并在搬运过程中进行随路Padding |
| [move_mask_reg](./move_mask_reg) | 本样例演示了SIMD场景下，基于RegBase编程范式下数据从UB到MaskReg之间的搬入搬出。 |
| [move_successive_align](./move_successive_align) | 本样例演示了SIMD场景下，基于RegBase编程范式的连续对齐搬运算子的核函数直调方法，算子支持单核运行。 |
| [move_successive_unalign](./move_successive_unalign) | 本样例演示了SIMD场景下，基于RegBase编程范式的连续非对齐搬运算子的核函数直调方法，算子支持单核运行。 |
| [move_unsuccessive_align](./move_unsuccessive_align) | 本样例演示了SIMD场景下，基于RegBase编程范式的非连续对齐搬运算子的核函数直调方法，算子支持单核运行。 |