# 什么是Ascend C<a name="ZH-CN_TOPIC_0000002500621200"></a>

Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。您可以通过[Ascend C主页](https://www.hiascend.com/zh/ascend-c)了解更详细的内容。

Ascend C提供多层级API，满足多维场景算子开发诉求。

-   **语言扩展层 C API**：开放芯片完备编程能力，支持数组分配内存，一般基于指针编程，提供与业界一致的C语言编程体验。
-   **基础API**：基于Tensor进行编程的C++类库API，实现单指令级抽象，为底层算子开发提供灵活控制能力。
-   **高阶API**：封装单核公共算法，涵盖一些常见的计算算法（如卷积、矩阵运算等），显著降低复杂算法开发门槛。
-   **算子模板库**：基于模板提供算子完整实现参考，简化Tiling（切分算法）开发，支撑用户自定义扩展。
-   **Python前端：**[PyAsc编程语言](https://gitcode.com/cann/pyasc)基于Python原生接口，提供芯片底层完备编程能力，支持基于Python接口开发高性能Ascend C算子。

![](../figures/成长地图.png)

## 快速入门<a name="zh-cn_topic_0000002484241782_section11489186191713"></a>

[![](../figures/zh-cn_image_0000002533161249.png)](./快速入门/)

## 成长地图<a name="zh-cn_topic_0000002484241782_section181811514171911"></a>

<div style="position:relative;display:inline-block;max-width:992px;width:100%;">
  <img src="../figures/zh-cn_image_0000002533161253.png" style="display:block;max-width:992px;width:100%;" />
  <a href="环境准备.md" title="环境准备" style="position:absolute;left:17.3%;top:10.3%;width:9.3%;height:3.9%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/编程模型/AI-Core-SIMD编程/抽象硬件架构.md" title="SIMD编程模型" style="position:absolute;left:17.8%;top:26.9%;width:10.9%;height:4.1%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/编程模型/AI-Core-SIMT编程/抽象硬件架构.md" title="SIMT编程模型" style="position:absolute;left:30.4%;top:27.3%;width:11.9%;height:3.3%;display:block;border:2px solid transparent;"></a>
  <a href="../算子实践参考/SIMD算子实现/矢量编程/概述.md" title="矢量编程" style="position:absolute;left:18.8%;top:47.6%;width:8.9%;height:3.5%;display:block;border:2px solid transparent;"></a>
  <a href="../算子实践参考/SIMD算子实现/矩阵编程（高阶API）/基础知识.md" title="基于高阶API" style="position:absolute;left:29.7%;top:47.8%;width:9.0%;height:2.9%;display:block;border:2px solid transparent;"></a>
  <a href="../算子实践参考/SIMD算子实现/矩阵编程（基础API）/分离模式.md" title="基于基础API" style="position:absolute;left:40.6%;top:48.0%;width:8.9%;height:3.4%;display:block;border:2px solid transparent;"></a>
  <a href="../算子实践参考/SIMD算子实现/融合算子编程/CV融合/基础知识.md" title="融合算子编程" style="position:absolute;left:51.2%;top:48.2%;width:8.3%;height:2.7%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/编译与运行/AI-Core-SIMD编译/算子编译简介.md" title="算子编译" style="position:absolute;left:64.1%;top:48.2%;width:8.6%;height:3.4%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/编译与运行/算子运行.md" title="Kernel直调" style="position:absolute;left:79.8%;top:43.4%;width:12.1%;height:6.0%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/调试调优/功能调试/CPU域孪生调试.md" title="CPU域调试" style="position:absolute;left:17.9%;top:69.7%;width:8.0%;height:3.8%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/调试调优/功能调试/NPU域上板调试.md" title="prinf/DumpTensor" style="position:absolute;left:30.6%;top:70.1%;width:12.8%;height:3.1%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/调试调优/功能调试/NPU域上板调试.md#section931475414217" title="msSanitizer" style="position:absolute;left:45.9%;top:69.7%;width:11.1%;height:4.3%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/调试调优/功能调试/NPU域上板调试.md#section2072113416285" title="msDebug工具" style="position:absolute;left:58.1%;top:69.5%;width:10.9%;height:4.4%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/调试调优/性能调优.md" title="msProf工具" style="position:absolute;left:77.8%;top:65.5%;width:18.0%;height:3.5%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/编译与运行/算子运行.md" title="Kernel直调" style="position:absolute;left:19.2%;top:88.5%;width:9.3%;height:3.8%;display:block;border:2px solid transparent;"></a>
  <a href="../编程指南/附录/AI框架算子适配/PyTorch框架.md" title="Pytorch框架" style="position:absolute;left:71.7%;top:87.1%;width:16.2%;height:6.4%;display:block;border:2px solid transparent;"></a>
</div>

## 概念原理<a name="zh-cn_topic_0000002484241782_section126809373191"></a>

[![](../figures/zh-cn_image_0000002533161259.png)](../编程指南/概念原理和术语/)

## API参考<a name="zh-cn_topic_0000002484241782_section116808372193"></a>

[![](../figures/zh-cn_image_0000002533161263.png)](../../api)

## 算子实践参考<a name="zh-cn_topic_0000002484241782_section16680133716195"></a>

[![](../figures/zh-cn_image_0000002533161267.png)](../算子实践参考/)

Ascend C支持在如下AI处理器型号使用：

-   Ascend 950PR/Ascend 950DT
-   Atlas A3 训练系列产品/Atlas A3 推理系列产品
-   Atlas A2 训练系列产品/Atlas A2 推理系列产品

-   Atlas 200I/500 A2 推理产品
-   Atlas 推理系列产品
-   Atlas 训练系列产品
