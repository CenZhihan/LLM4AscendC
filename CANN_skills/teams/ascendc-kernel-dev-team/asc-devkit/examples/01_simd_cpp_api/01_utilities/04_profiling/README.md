# Profiling样例介绍
## 概述
本样例展示了如何通过Profiling工具采集性能数据。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [msProf](./msProf) | 本样例基于Ascend C编程语言实现了MatmulLeakyRelu样例, 同时使用msProf工具调试，给出了对应的端到端实现 |
| [torch_library_report_tensor](./torch_library_report_tensor) | 本样例展示了通过<<<>>>内核调用符调用核函数时，如何集成Profiling并采集Add算子的Shape信息。 |