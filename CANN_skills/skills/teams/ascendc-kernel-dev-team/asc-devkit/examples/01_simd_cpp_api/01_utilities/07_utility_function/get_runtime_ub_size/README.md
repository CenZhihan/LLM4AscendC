# GetRuntimeUBSize样例

## 概述

本样例基于GetRuntimeUBSize获取运行时UB的大小（单位为Byte），开发者可以根据UB大小计算循环次数等参数值。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── get_runtime_ub_size
│   ├── CMakeLists.txt                       // 编译工程文件
│   └── get_ub_size_in_bytes_custom.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  根据UB大小以及输入数据总量来计算op.process()函数的循环次数。
- 算子规格：  
  <table>
  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">126976</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">z</td><td align="center">126976</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_get_runtime_ub_size</td></tr>
  </table>
  </table>

- 算子实现：  
  - kernel实现  

    本算子中Init中通过调用GetRuntimeUBSize获取UB的大小，从而计算出Process被调用的次数。Process的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm搬运至Local Memory，存储在xLocal中。Compute任务负责对xLocal求绝对值，结果存储在outLocal中。CopyOut任务负责将输出数据从outLocal搬运至Global Memory上的输出Tensor outGm中。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

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

  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
