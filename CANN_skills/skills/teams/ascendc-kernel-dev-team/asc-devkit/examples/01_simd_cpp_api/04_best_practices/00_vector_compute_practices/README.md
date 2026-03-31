# Vector Compute Practices样例介绍
## 概述
本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，优化性能，使用double buffer进行流水排布，支持main函数和kernel函数在同一个cpp文件中实现，并提供<<<>>>直调方法。
 
## 算子开发样例
|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [add_doublebuffer](./add_doublebuffer) | 本样例介绍基于静态Tensor方式编程的场景下Add算子的实现方法，优化性能，使用double buffer进行流水排布，支持main函数和kernel函数在同一个cpp文件中实现，并提供<<<>>>直调方法。 |
| [l2_cache_bypass](./l2_cache_bypass) | 本样例介绍了设置L2 CacheMode的方法以及其影响场景，并提供核函数直调方法。 |