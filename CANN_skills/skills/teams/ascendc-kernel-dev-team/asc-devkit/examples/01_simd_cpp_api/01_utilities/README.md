# Utilities样例介绍
## 概述
基于Ascend C的简单样例，通过printf、assert、debug等API介绍上板打印、异常检测、CPU孪生调试等系统工具使用方法，适用于调试阶段。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [00_printf](./00_printf) | 本样例通过Ascend C编程语言实现了Printf算子，展示了上板打印功能 |
| [01_assert](./01_assert) | 本样例通过Ascend C编程语言实现了Assert算子，展示了异常检测功能 |
| [02_dump](./02_dump) | 本样例通过Ascend C编程语言实现了asc_dump算子，展示了数据调测功能 |
| [03_cpudebug](./03_cpudebug) | 本样例通过Ascend C编程语言实现了Add算子的CPU Debug调测，给出了对应的端到端实现 |
| [05_mindstudio_tools](./05_mindstudio_tools) | 本样例基于MindStudio工具，包含msProf和msSanitizer两个调测工具的使用样例 |
| [06_clock](./06_clock) | 本样例通过Ascend C编程语言实现了Gather算子，同时在算子中添加clock调测，给出了对应的端到端实现 |
| [07_utility_function](./07_utility_function) | 本样例介绍了Ascend C的工具函数API，包括GetRuntimeUBSize和GetUBSizeInBytes等 |
