# 自定义算子工程多 Vendor 并行编译、打包和部署样例

## 概述

本样例展示如何在一个顶层 CMake 工程中，使用 `ExternalProject_Add` 并行编译两个独立自定义算子工程：
- `add_custom`（AddCustom）
- `leaky_relu_custom`（LeakyReluCustom）

每个子工程会分别完成自定义算子的编译、打包，并生成独立的 `custom_opp_*.run` 安装包。

## 支持的产品

本样例支持如下产品型号：
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas 推理系列产品

> 注意: 本样例中涉及多个算子示例，请以各个算子示例实际支持的产品型号为准。

## 目录结构

```text
parallel_ops_package
├── CMakeLists.txt
├── README.md
├── add_custom
│   ├── CMakeLists.txt
│   ├── framework
│   │   ├── CMakeLists.txt
│   │   └── tf_plugin
│   │       ├── CMakeLists.txt
│   │       └── tensorflow_add_custom_plugin.cc
│   ├── op_host
│   │   ├── CMakeLists.txt
│   │   └── add_custom
│   │       └── add_custom_host.cpp
│   └── op_kernel
│       ├── CMakeLists.txt
│       └── add_custom
│           ├── add_custom_kernel.cpp
│           └── add_custom_tiling.h
└── leaky_relu_custom
    ├── CMakeLists.txt
    ├── framework
    │   ├── CMakeLists.txt
    │   └── onnx_plugin
    │       ├── CMakeLists.txt
    │       └── leaky_relu_custom_plugin.cc
    ├── op_host
    │   ├── CMakeLists.txt
    │   └── leaky_relu_custom
    │       └── leaky_relu_custom_host.cpp
    └── op_kernel
        ├── CMakeLists.txt
        └── leaky_relu_custom
            ├── leaky_relu_custom_kernel.cpp
            └── leaky_relu_custom_tiling.h
```

## 样例描述

`parallel_ops_package` 与 `custom_op` 使用相同的 Add/LeakyRelu 样例描述，本文不重复维护，请参考：

- [custom_op/README.md 的“样例描述”章节](../custom_op/README.md#样例描述)

## 样例规格描述

`parallel_ops_package` 与 `custom_op` 使用相同的 Add/LeakyRelu 规格，本文不重复维护规格表，请参考：

- [custom_op/README.md 的“样例规格描述”章节](../custom_op/README.md#样例规格描述)

## 代码实现介绍

`parallel_ops_package` 的算子实现与 `custom_op` 保持一致，本文不重复维护实现细节，请参考：

- [custom_op/README.md 的“代码实现介绍”章节](../custom_op/README.md#代码实现介绍)

## 编译运行

在本样例根目录下执行如下步骤，编译、打包并部署自定义样例包。

- 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 编译、打包样例并部署两个算子包

  ```bash
  cmake -S . -B build
  cmake --build build -j
  # add_custom 包
  ./build/add_custom/custom_opp_*.run

  # leaky_relu_custom 包
  ./build/leaky_relu_custom/custom_opp_*.run
  ```

  执行结果如下，说明执行成功。

  ```log
  SUCCESS
  ```

## 构建结果说明

顶层工程会在 `build/` 下生成两个子目录：
- `build/add_custom/`：AddCustom 的中间产物与安装包
- `build/leaky_relu_custom/`：LeakyReluCustom 的中间产物与安装包

这两个目录互相独立，便于多 Vendor 场景下并行开发与发布。
