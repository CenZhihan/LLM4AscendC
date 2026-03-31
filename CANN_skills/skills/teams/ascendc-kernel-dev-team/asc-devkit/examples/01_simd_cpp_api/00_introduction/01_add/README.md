# Add样例介绍

## 概述

本样例介绍了基于Ascend C的Add自定义Vector算子的核函数直调方法，实现两个输入张量的逐元素相加，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [basic_api_memory_allocator_add](./basic_api_memory_allocator_add) | 样例基于静态Tensor方式编程实现Add样例，展示了LocalMemAllocator进行线性内存分配 |
| [basic_api_tque_add](./basic_api_tque_add) | 样例以Add算子为例，采用TQue内存管理机制实现数据搬运与计算任务的协同调度 |