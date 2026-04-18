# 标量计算类api样例介绍

## 概述

本路径下包含了与标量计算相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

| 目录名称 | 功能描述 |
| ------- | -------- |
| [read_gm_by_pass_dcache](./read_gm_by_pass_dcache)   | 本样例介绍基础api ReadGmByPassDcache的调用，该api功能：不经过DCache从GM地址上读数据。 |
| [write_gm_by_pass_dcache](./write_gm_by_pass_dcache) | 本样例介绍基础api WriteGmByPassDcache的调用，该api功能：不经过DCache向GM地址上写数据。 |