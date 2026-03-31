# 工具函数类api样例介绍

## 概述

本路径下包含了与工具函数相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

| 目录名称                                 |  功能描述                                           |
|--------------------------------------| ------------------------------------------------- |
| [get_runtime_ub_size](./get_runtime_ub_size)   | 本样例基于GetRuntimeUBSize获取运行时UB的大小（单位为Byte），开发者可以根据UB大小计算循环次数等参数值。 |
| [get_ub_size_in_bytes](./get_ub_size_in_bytes)   | 本样例基于GetUBSizeInBytes获取UB的大小（单位为Byte），开发者可以根据UB大小计算循环次数等参数值。 |