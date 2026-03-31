# Compatibility Guide样例介绍

## 概述

针对不兼容Atlas A2 训练系列产品/Atlas A2 推理系列产品和Ascend 950PR/Ascend 950DT的部分特性，本小节提供了若干样例，用户可以根据样例进行迁移。
本小节样例使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 算子开发样例

|  目录名称  |  功能描述  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [data_copy_l1togm_910B](./data_copy_l1togm_910B) | 实现L1到GM搬运的端到端用例，直接使用DataCopy接口进行搬运。 |
| [data_copy_l1togm_950](./data_copy_l1togm_950) | 新架构下实现L1到GM搬运的端到端用例，DataCopy接口不支持L1 Buffer到GM的通路。用户在cube only场景下，可以在GM多分配一个单位矩阵，通过Mmad矩阵乘法计算输出到L0C Buffer，再从L0C Buffer通过Fixpipe搬运到GM。 |
| [fill_910B](./fill_910B) | 本样例直接使用基础API InitConstValue对L0A Buffer和L0B Buffer进行初始化。 |
| [fill_950](./fill_950) | 新架构版本删除L0A Buffer/L0B Buffer初始化的相关硬件指令。用户可以通过先初始化L1 Buffer，再通过LoadData接口将L1 Buffer上的数据搬运到L0A Buffer/L0B Buffer。 |
| [matmul_s4_910B](./matmul_s4_910B) | 本样例直接使用Matmul高阶APi进行矩阵计算。 |
| [matmul_s4_950](./matmul_s4_950) | 新架构下Cube计算单元删除int4b_t数据类型。用户可以在算子侧通过MIX模式再Vector Core进行int4b_t到int8_t的Cast转换，再通过UB搬运到L1后进行Mmad计算。 |
| [pattern_transformation](./pattern_transformation) | 新架构下基础的mmad样例，从L1 Buffer->L0A Buffer的通路不需要做Nz2Zz分型转换。 |
| [scatter](./scatter) | 本样例介绍兼容Scatter算子实现及核函数直调方法。 |
| [set_loaddata_boundary_910B](./set_loaddata_boundary_910B) | 本样例使用SetLoadDataBoundary接口设置L1 Buffer的边界值。 |
| [set_loaddata_boundary_950](./set_loaddata_boundary_950) | 新架构硬件删除了L1 Buffer的边界值设定相关寄存器，不再支持SetLoadDataBoundary接口。该接口用于设置Load3x时L时Buffer的边界值。如果指令在处理源操作数时，源操作数在L1 Buffer上的地址超出设置的边界，则会从L1 Buffer的起始地址开始夺取。设置为0表示无边界，可使用整个L1 Buffer。 |