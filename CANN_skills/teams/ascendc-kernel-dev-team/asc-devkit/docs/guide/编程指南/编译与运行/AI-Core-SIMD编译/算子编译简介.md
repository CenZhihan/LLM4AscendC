# 算子编译简介<a name="ZH-CN_TOPIC_0000002457558450"></a>

本章节介绍的算子编译方法支持开发者通过bisheng命令行和CMake进行手动配置编译选项，或编写CMake脚本来实现编译。开发者可以将Host侧main.cpp和Device侧Kernel核函数置于同一实现文件中，以实现异构编译。

-   目前，该编译方法仅支持如下型号：
    -   Ascend 950PR/Ascend 950DT
    -   Atlas A3 训练系列产品/Atlas A3 推理系列产品
    -   Atlas A2 训练系列产品/Atlas A2 推理系列产品
    -   Atlas 推理系列产品

-   异构编译场景中的编程相关约束请参考[约束说明](约束说明.md)。

