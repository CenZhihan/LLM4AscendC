# 静态Tensor编程样例

## 概述

本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，并提供核函数直调方法。

## 算子开发样例

|  目录名称 |  功能描述  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [add_custom_v1](./add_custom_v1) | 使用静态Tensor编程方法进行add算子的编程，未使用优化能力。 |
| [add_custom_v2](./add_custom_v2) | 使用静态Tensor编程方法进行add算子的编程，优化性能，使用double buffer进行流水排布。 |
| [add_custom_v3](./add_custom_v3) | 使用静态Tensor编程方法进行add算子的编程，优化double buffer实现，简化判断逻辑，并使用LocalMemAllocator简化代码。 |
| [add_custom_v4](./add_custom_v4) | 使用静态Tensor编程方法进行add算子的编程，基于add_custom_v3，修改地址分配逻辑，消除bank冲突。 |
| [add_custom_v5](./add_custom_v5) | 使用静态Tensor编程方法进行add算子的编程，实现SetFlag/WaitFlag同步指令的循环内外依赖。 |
| [add_custom_v6](./add_custom_v6) | 使用静态Tensor编程方法进行add算子的编程，实现SetFlag/WaitFlag同步指令的If-Else内外依赖。 |