# 自定义算子tiling下沉图模式调用样例

## 概述

本样例基于示例自定义算子工程，介绍了PyTorch图模式下调用自定义算子，并通过使能Tiling下沉到device侧执行，优化调度性能。

## 支持的产品

本样例支持如下产品型号：
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── tiling_sink_programming
│   └── test_add_custom_tiling_sink.py
```

## 代码实现介绍

样例脚本`test_add_custom_tiling_sink.py`包含三个关键部分：
1. 在PyTorch中注册自定义算子，并为Meta/CPU/PrivateUse1提供占位实现，保证可入图。
2. 注册FX到GE的转换器，将`add_custom_tiling_sink`映射为GE侧自定义算子`AddCustomTilingSink`。
3. 通过`torch.compile`启用图模式执行，并打开`tiling_schedule_optimize`配置。

## 编译运行

- 安装PyTorch以及Ascend Extension for PyTorch插件

  请参考[pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)开源代码仓或[Ascend Extension for PyTorch昇腾社区](https://hiascend.com/document/redirect/Pytorch-index)的安装说明，选取支持的`Python`版本配套发行版，完成`torch`和`torch-npu`的安装。

- 编译、打包和部署自定义算子工程

  运行此样例前，需先进入[自定义算子工程样例](../../00_compilation/custom_op/)目录完成编译、打包和部署。

- 安装前置依赖

  ```bash
  pip3 install expecttest
  ```

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

- 样例执行

  在本样例根目录下执行如下步骤，运行该样例。

  ```bash
  python3 test_add_custom_tiling_sink.py
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  Ran 1 test in **s
  OK
  ```