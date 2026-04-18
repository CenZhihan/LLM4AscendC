
# 原子操作类api样例介绍

## 概述

本路径下包含了与原子操作相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

| 目录名称                                 |  功能描述                                           |
|--------------------------------------| ------------------------------------------------- |
| [set_atomic_add](./set_atomic_add)   | 本样例基于SetAtomicAdd为后续的从VECOUT/L0C/L1到GM的数据传输开启原子累加。 |
| [set_atomic_max](./set_atomic_max)   | 本样例基于SetAtomicMax设置后续从VECOUT传输到GM的数据原子比较，可用于将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。 |
| [set_atomic_min](./set_atomic_min)   | 本样例基于SetAtomicMin设置后续从VECOUT传输到GM的数据原子比较，可用于将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。 |
| [set_atomic_type](./set_atomic_type) | 本样例基于SetAtomicType为原子操作设定不同的数据类型。 |