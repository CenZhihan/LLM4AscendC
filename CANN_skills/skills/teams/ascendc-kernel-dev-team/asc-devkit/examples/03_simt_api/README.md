# 样例运行验证

开发者基于Ascend C SIMT编程实现自定义算子后，可通过单算子调用的方式验证算子功能。本代码仓提供部分算子实现及其调用样例，具体如下。

## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------- | ------------------------------------- |
| [00_introduction](./00_introduction) | 基于Ascend C SIMT编程的简单的示例，通过Ascend C编程语言实现了自定义算子，分别给出对应的<<<>>>直调实现，适合初学者 |
| [01_utilities](./01_utilities) | 基于Ascend C SIMT的调试调优样例，通过printf、assert、clock等API介绍上板打印、打点、CPU孪生调试等系统工具使用方法，适用于调试阶段 |
| [02_features](./02_features) | 展示Ascend C SIMT编程的重点特性的样例，包含基础编程、内存资源管理、线程同步、原子操作等特性。 |

## npu-arch编译选项说明

开发者需根据实际的执行环境，修改具体样例目录下CMakeLists.txt文件中的--npu-arch编译选项，参考下表中的对应关系，修改为环境对应的npu-arch参数值。
| 产品型号 |  npu-arch参数 |
| ---- | ---- |
| Ascend 950PR/Ascend 950DT | --npu-arch=dav-3510 |