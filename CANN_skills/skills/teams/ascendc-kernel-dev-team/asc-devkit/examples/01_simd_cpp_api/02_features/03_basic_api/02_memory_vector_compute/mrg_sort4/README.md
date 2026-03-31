# MrgSort4样例

## 概述

本样例介绍基础api MrgSort4的调用，该api的功能：将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序。

## 支持的产品

- Atlas 推理系列产品AI Core

## 目录结构介绍

```
├── mrg_sort4
│   ├── CMakeLists.txt          // 编译工程文件
│   └── mrg_sort4.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序。

- 算子规格：  
  <table>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">64</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">512</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_mrg_sort4</td></tr>  
  </table>

- 算子实现：  
  本样例中实现的是MrgSort4算子。

  - Kernel实现  
    MrgSort4算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责创建4个Region Proposals并完成排序，然后调用MrgSort4 api将4个Region Proposals按照score从大到小合并为一个，并将结果存储到dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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

  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
