# AICPU样例介绍

## 概述

本样例介绍了使用AI CPU算子进行Tiling下沉计算的实现方法。

## 算子开发样例

| 目录名称 | 功能描述 |
| ------- | -------- |
| [aicpu_device_tiling](./aicpu_device_tiling) | 本样例介绍了使用AI CPU算子进行Tiling下沉计算的实现方法 |
| [tiling_sink_programming](./tiling_sink_programming) | 本样例基于示例自定义算子工程，介绍了PyTorch图模式下调用自定义算子，并通过使能Tiling下沉到device侧执行，优化调度性能 |