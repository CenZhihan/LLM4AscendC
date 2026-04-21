# 激活函数算子样例介绍

## 概述

本样例集展示了激活函数高阶 API 的典型用法，每个样例包含完整的端到端实现。

## 算子开发样例

|  目录名称                                                   |  功能描述                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [adjustsoftmaxres](./adjustsoftmaxres) | 本样例介绍了调用AdjustSoftMaxRes高阶API实现AdjustSoftMaxRes算子，并按照核函数直调的方式给出了对应的端到端实现 |
| [geglu](./geglu) | 本样例演示了基于GeGLU高阶API的算子实现。GeGLU是一种结合GELU激活函数的GLU变体 |
| [gelu](./gelu) | 本样例演示了基于Gelu高阶API的算子实现。样例对输入Tensor按元素做GELU激活计算 |
| [logsoftmax](./logsoftmax) | 本样例演示了基于LogSoftMax高阶API的算子实现。样例对输入tensor做LogSoftmax计算 |
| [reglu](./reglu) | 本样例演示了基于ReGLU高阶API的算子实现。ReGlu是一种GLU变体，使用Relu作为激活函数 |
| [sigmoid](./sigmoid) | 本样例演示了基于Sigmoid高阶API实现的算子实现。样例按元素做逻辑回归Sigmoid |
| [silu](./silu) | 本样例演示了基于Silu高阶API的算子实现。样例按元素做Silu运算 |
| [simplesoftmax](./simplesoftmax) | 本样例介绍了调用SimpleSoftMax高阶API实现softmax单算子，使用计算好的sum和max数据对输入tensor做Softmax计算 |
| [softmax](./softmax) | 本样例介绍了调用SoftMax高阶API实现softmax单算子，将输入tensor[m0, m1, ...mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n] |
| [softmaxflashv2](./softmaxflashv2) | 本样例介绍了调用SoftmaxFlashV2高阶api实现softmaxflashv2单算子，SoftmaxFlash增强版本，对应FlashAttention-2算法 |
| [softmaxflashv3](./softmaxflashv3) | 本样例介绍了调用SoftmaxFlashV3高阶api实现softmaxflashv3单算子，SoftmaxFlash增强版本，对应Softmax PASA算法 |
| [softmaxgrad](./softmaxgrad) | 本样例介绍了调用SoftmaxGrad高阶API实现softmaxgrad单算子，将输入tensor[m0, m1, ...mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n] |
| [softmaxgradfront](./softmaxgradfront) | 本样例介绍了调用SoftmaxGradFront高阶API实现softmaxgradfront单算子，将输入tensor[m0, m1, ...mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n] |
| [swiglu](./swiglu) | 本样例演示了基于SwiGLU高阶API实现的算子实现。样例采用Swish作为激活函数的GLU变体 |
| [swish](./swish) | 本样例演示了基于Swish高阶API的算子实现。样例按元素做Swish激活计算 |