# Template Selector样例介绍

## 概述

本样例介绍了Tiling策略和Tiling模板编程的实现方法。

## 算子开发样例

| 目录名称 | 功能描述 |
| ------- | -------- |
| [tiling_strategy](./tiling_strategy) | 本样例实现了一个支持多种数据类型的向量加法算子，采用精细的多核并行数据切分（Tiling）策略，以实现高效能计算 |
| [tiling_template_programming](./tiling_template_programming) | 本样例基于示例自定义算子工程，使用Tiling模板编程进行单算子API方式的算子执行，以有效减少多TilingKey的复杂度 |