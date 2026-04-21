# 自定义算子工程+aclnn单算子API调用样例

## 概述

本样例基于示例自定义算子工程，介绍了aclnn`OpType`单算子API的方式执行固定shape算子。

## 支持的产品

本样例支持如下产品型号：
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 200I/500 A2 推理产品
- Atlas 推理系列产品

## 目录结构介绍

```
├── aclnn_invocation
│   ├── CMakeLists.txt
│   └── main.cpp
```

## 代码实现介绍

完成自定义算子的开发部署后，可以通过单算子API调用的方式来验证单算子的功能。具体可参考[单算子API调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)章节以及[单算子调用](https://hiascend.com/document/redirect/CannCommunityCppOpcall)中“单算子API执行”相关章节。

单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，可直接调用单算子API接口。

本样例中通过`aclnnAddCustomGetWorkspaceSize`与`aclnnAddCustom`两段式接口执行算子计算，核心流程如下：
1. 创建输入/输出`aclTensor`并准备Device侧数据。
2. 调用`aclnnAddCustomGetWorkspaceSize`获取本次计算所需workspace大小，并申请对应Device内存。
3. 调用`aclnnAddCustom`执行计算，同步流`aclrtSynchronizeStream`，并将结果拷回Host侧进行验证。

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

  在本样例根目录下执行如下步骤，运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake .. && make -j
  ./execute_add_op
  ```

  执行结果如下，说明执行成功。

  ```log
  test pass
  ```
