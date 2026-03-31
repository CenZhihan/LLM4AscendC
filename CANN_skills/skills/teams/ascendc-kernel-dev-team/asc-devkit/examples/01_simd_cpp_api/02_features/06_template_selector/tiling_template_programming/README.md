# 自定义算子工程+aclnn单算子API调用样例（Tiling模板编程）

## 概述

本样例基于示例自定义算子工程，使用Tiling模板编程进行单算子API方式的算子执行，以有效减少多TilingKey的复杂度。

## 支持的产品

本样例支持如下产品型号：
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── tiling_template_programming
│   ├── CMakeLists.txt
│   └── main.cpp
```

## 代码实现介绍

完成自定义算子的开发部署后，可以通过单算子API调用的方式来验证单算子的功能。具体可参考[单算子API调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)章节以及[单算子调用](https://hiascend.com/document/redirect/CannCommunityCppOpcall)中“单算子API执行”相关章节。

Tiling模板编程能够有效减少因为TilingKey数量众多，导致的难以记忆和理解的问题，提升算子编程易用性。它具有C++同等的泛型编程、零成本抽象、编译期计算等特征。具体可参考[Ascend C算子开发](http://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“编程指南”-“附录”-“工程化算子开发”-“Host侧Tiling实现”-“Tiling模板编程”相关章节。

## 编译运行

- 编译、打包和部署自定义算子工程

  运行此样例前，需先进入[自定义算子工程样例](../../00_compilation/custom_op/)目录完成编译、打包和部署。

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

  在本样例根目录下执行如下步骤，最后使用形如`./execute_add_template_op [float|float16] [128|2048]`运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake .. && make -j
  ./execute_add_template_op float16 128
  ./execute_add_template_op float16 2048
  ./execute_add_template_op float 128
  ./execute_add_template_op float 2048
  ```

  执行结果如下，说明执行成功。


  ```log
  test pass
  ```
