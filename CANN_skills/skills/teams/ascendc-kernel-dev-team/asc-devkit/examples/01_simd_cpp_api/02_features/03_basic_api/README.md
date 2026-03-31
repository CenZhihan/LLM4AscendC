# Basic API样例介绍

## 概述

本样例介绍了Ascend C Basic API的使用方法，包括数据搬运、矩阵计算、内存向量计算、资源管理、同步控制、系统访问、原子操作和标量计算等。

## 算子开发样例

| 目录名称  | 功能描述 |
| --------- | --------- |
| [00_data_movement](./00_data_movement/) | 本样例路径以copy、data_copy_pad、data_copy_ub2l1等算子为示例，展示了数据搬运接口的使用。|
| [01_matrix_compute](./01_matrix_compute/) | 本样例路径以batch_mmad、fixpipe_co12c1_quantization等算子为示例，展示了矩阵计算接口的使用。|
| [02_memory_vector_compute](./02_memory_vector_compute/) | 本样例路径以reduce、sort、transpose等算子为示例，展示了内存向量计算接口的使用。|
| [04_resource_management](./04_resource_management/) | 本路径下包含了与资源管理相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [06_sync_control](./06_sync_control/) | 本路径下包含了与同步控制相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [07_system_access](./07_system_access/) | 本路径下包含了与系统变量访问相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [08_atomic_operations](./08_atomic_operations/) | 本路径下包含了与原子操作相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |
| [09_scalar_computation](./09_scalar_computation/) | 本路径下包含了与标量计算相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。         |