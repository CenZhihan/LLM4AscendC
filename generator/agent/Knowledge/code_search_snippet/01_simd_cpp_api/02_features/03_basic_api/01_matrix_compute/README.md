# Matrix Compute API样例介绍
## 概述
本样例集介绍了Matmul API不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例
| 目录名称                                                                                       |  功能描述                                              |
|--------------------------------------------------------------------------------------------| ---------------------------------------------------- |
| [fixpipe_co12c1_quantization_s322f16](./fixpipe_co12c1_quantization_s322f16)               | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入C1，并使能随路量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_co12c1_quantization_s322s8](./fixpipe_co12c1_quantization_s322s8)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitflag能力组合，将矩阵乘结果由int32_t类型数据量化为int8_t类型 |
| [fixpipe_co12c1_quantization_f322s8](./fixpipe_co12c1_quantization_f322s8)                 | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitflag能力组合，输入half类型数据，输出由float类型量化为int8_t类型, int8_t输出默认开启channel merge能力。 |
| [fixpipe_co12c1_quantization_s322f16_v2](./fixpipe_co12c1_quantization_s322f16_v2)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitflag能力组合，输入int8_t类型数据，输出由int32_t类型量化为half类型。 |
| [fixpipe_co12c1_tensor_quantization_s322f16](./fixpipe_co12c1_tensor_quantization_s322f16) | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入C1，并使能随路tensor量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_co12gm_tensor_quantization_s322f16](./fixpipe_co12gm_tensor_quantization_s322f16) | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入GM，并使能随路量化将矩阵乘结果由int32_t类型数据量化为half类型 |
| [fixpipe_co12gm_quantization_f322f16](./fixpipe_co12gm_quantization_f322f16)               | 本样例介绍如何使用基础API Fixpipe将矩阵乘的结果从CO1搬入GM，并使能随路量化将矩阵乘结果由float类型数据量化为half类型 |
| [fixpipe_co12gm_quantization_f322s8](./fixpipe_co12gm_quantization_f322s8)                 | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，并支持随路NZ2ND，unitflag与随路quant, relu能力组合，输入half类型数据，输出由float类型量化为int8_t类型 |
| [fixpipe_co12gm_quantization_s322s8](./fixpipe_co12gm_quantization_s322s8)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，并支持随路NZ2ND，unitflag与随路quant, relu能力组合，将矩阵乘结果由int32_t类型量化为int8_t类型 |
| [fixpipe_co12gm_channelsplit](./fixpipe_co12gm_channelsplit)                 | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，float类型NZ输出下，使能channel split特性，将L0c上16x16小z分型矩阵拆分成两个独立的16X8小z分型矩阵输出到GM |
| [fixpipe_co12gm_quantization_s322f16](./fixpipe_co12gm_quantization_s322f16)               | 本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到GM，并支持随路NZ2ND，unitflag与随路quant, relu能力组合，输入int8_t类型数据，输出由int32_t类型量化为half类型 |
| [fixpipe_nz2dn_tensor_quantization_f322f16](./fixpipe_nz2dn_tensor_quantization_f322f16)   | 本样例介绍如何使用基础API的Fixpipe将矩阵乘的结果从CO1搬入GM，完成NZ2DN分形转换，并使能随路量化将矩阵乘结果由float类型数据量化为half类型 |
| [load_data](./load_data)                                                                   | 本样例介绍基于基础API LoadData实现A1至A2和B1至B2的数据搬运，其中A1至A2使用Load3D搬运，B1至B2使用Load2D搬运 |
| [load_data_with_transpose_b8](./load_data_with_transpose_b8)                               | 本样例介绍基础API LoadDataWithTranspose b8数据类型下的使用 |
| [load_data_with_transpose_b16](./load_data_with_transpose_b16)                             | 本样例介绍基础API LoadDataWithTranspose b16数据类型下的使用 |
| [load_data_with_transpose_b32](./load_data_with_transpose_b32)                             | 本样例介绍基础API LoadDataWithTranspose b32数据类型下的使用 |
| [batch_mmad](./batch_mmad)                                                                 | 本样例介绍在输入为float数据类型并且左、右矩阵均不转置的场景下，带batch的矩阵乘法，其中从GM-->L1、L0C-->GM、L0C-->L1这三条通路分别采用了DataCopy ND2NZ和Fixpipe批量搬运数据，从L1-->L0A/L0B以及Mmad执行矩阵乘这两个步骤则是循环batch次，每次循环内只处理一对左、右矩阵 |
| [mmad](./mmad)                                                                             | 本样例介绍基于基础API Mmad实现矩阵乘 |
| [mmad_load3dv2](./mmad_load3dv2)                                                           | 本样例介绍LoadData3DV2指令将A、B矩阵从L1搬运到L0A/L0B的过程，其中 A 和 B 分别表示矩阵乘法的左右输入矩阵。LoadData3DV2指令参数配置及执行指令前后各个矩阵数据排布变化，均配合示意图进行了说明 |
| [mmad_s8_f16_f32_with_A_B_transpose_option](./mmad_s8_f16_f32_with_A_B_transpose_option)                                                           | 本样例介绍了在 int8_t / half / float 三种数据类型下，以及左、右矩阵均不转置 / 左矩阵不转置、右矩阵转置 / 左矩阵转置、右矩阵不转置 / 左、右矩阵均转置 共 12 种矩阵乘法场景中，相关指令的使用方法，其中 A 和 B 分别表示矩阵乘法的左右输入矩阵。|
| [mmad_unitflag](./mmad_unitflag)                                                                             | 本样例介绍是否使能unitFlag对于Mmad指令执行矩阵乘法性能的影响。 |
| [mmad_with_bias](./mmad_with_bias)                                                         | 本样例介绍基于基础API Mmad实现带Bias的矩阵乘 |
| [mmad_with_sparse](./mmad_with_sparse)                                                     | 本样例介绍基础API MmadWithSparse调用 |
