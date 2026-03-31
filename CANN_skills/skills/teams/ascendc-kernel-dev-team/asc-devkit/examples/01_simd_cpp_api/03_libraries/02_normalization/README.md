# Normalization API样例介绍

## 概述

本样例集介绍了归一化操作不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [batchnorm](./batchnorm) | 本样例基于Kernel直调算子工程，介绍了调用BatchNorm高阶API实现batchnorm单算子，BatchNorm是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同，从而加速训练过程和提高模型的泛化能力（有效减少梯度消失和梯度爆炸问题） |
| [deepnorm](./deepnorm) | 本样例基于Kernel直调算子工程，介绍了调用DeepNorm高阶API实现deepnorm单算子，在深层神经网络训练过程中，执行层LayerNorm归一化时，可以使用DeepNorm进行替代，通过扩大残差连接来提高Transformer的稳定性 |
| [groupnorm](./groupnorm) | 本样例基于Kernel直调算子工程，介绍了调用GroupNorm高阶API实现groupnorm单算子，对输入tensor按行做Groupnorm计算 |
| [layernorm](./layernorm) | 本样例基于Kernel直调算子工程，介绍了调用LayerNorm高阶API实现layernorm单算子，对输入tensor按行做Layernorm计算 |
| [layernorm_grad](./layernorm_grad) | 本样例基于Kernel直调算子工程，介绍了调用LayerNormGrad高阶API实现LayernormGradCustom单算子，主要演示LayerNormGrad高阶API在Kernel直调工程中的调用 |
| [layernorm_v2](./layernorm_v2) | 本样例基于Kernel直调算子工程，介绍了调用LayerNorm高阶API实现layernorm单算子，对输入tensor按行做Layernorm计算，得到Mean，Rstd和最后的归一化结果 |
| [layernormgradbeta](./layernormgradbeta) | 本样例基于Kernel直调算子工程，介绍了调用LayerNormGradBeta高阶API实现layernormgradbeta单算子，LayerNormGradBeta是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同，从而加速训练过程和提高模型的泛化能力（有效减少梯度消失和梯度爆炸问题） |
| [normalize](./normalize) | 本样例基于Kernel直调算子工程，介绍了调用Normalize高阶API实现normalize单算子，LayerNorm中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和y |
| [rmsnorm](./rmsnorm) | 本样例基于Kernel直调算子工程，介绍了调用RmsNorm高阶API实现rmsnorm单算子，实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化 |
| [welford_finalize](./welford_finalize) | 本样例演示了基于WelfordFinalize高阶API的算子实现，Welford计算是一种在线计算均值和方差的方法 |
| [welford_update](./welford_update) | 本样例基于Kernel直调算子工程，介绍了调用WelfordUpdate高阶API实现welford_update单算子，Welford是一种在线计算均值和方差的方法 |