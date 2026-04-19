
# 资源管理类api样例介绍

## 概述

本路径下包含了与资源管理相关的多个API的样例。每个样例均基于Ascend C的<<<>>>直调方法，支持main函数和kernel函数在同一个cpp文件中实现。

## 算子开发样例

|  目录名称                                                  |  功能描述                                             |
| ----------------------------------------------------------- | --------------------------------------------------- |
| [tpipe_reuse](./tpipe_reuse) | 本样例基于TPipe::Init和TPipe::Destory，实现TPipe重复申请与使用。|
| [tpipe_buf_pool_reuse](./tpipe_buf_pool_reuse) | 本样例基于TPipe::InitBufPool初始化TBufPool内存资源池，适用于内存资源有限时，希望手动指定UB/L1内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。|
| [get_tpipe_ptr](./get_tpipe_ptr) | 样例基于GetTPipePtr获取全局TPipe指针，核函数无需显式传入TPipe指针，即可进行TPipe相关的操作。 |
| [tbufpool_buf_pool_subdivision](./tbufpool_buf_pool_subdivision) | 本样例基于TBufPool::InitBufPool进行TBufPool资源划分，适用于将Tpipe::InitBufPool接口划分的整块资源，继续划分成小块资源的场景。 |
| [customized_tbuf_pool](./customized_tbuf_pool) | 本样例基于EXTERN_IMPL_BUFPOOL宏来辅助用户自定义TBufPool，适用于以下场景：开发者有自定义的内存块分配需求，比如非连续内存块、内存块在不同TQue之间共享等。 |
| [add_dynamic](./add_dynamic) | 本样例演示基于动态Tensor编程模型的AddN算子实现，该实现采用ListTensorDesc结构处理多输入参数，结合TQue内存管理机制实现数据搬运与计算任务的协同调度 |
| [tmp_buffer](./tmp_buffer) | 本样例展示了一个支持bfloat16_t数据类型的向量加法（Add）算子，并重点演示了在算子计算过程中使用临时缓冲区（TmpBuf）进行数据转换的典型方法 |
| [static_tensor_programming](./static_tensor_programming) | 本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，并提供核函数直调方法 |