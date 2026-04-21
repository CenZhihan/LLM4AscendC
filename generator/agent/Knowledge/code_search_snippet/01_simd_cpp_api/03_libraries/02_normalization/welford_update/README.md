# WelfordUpdate样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用WelfordUpdate高阶API实现welford_update单算子，Welford是一种在线计算均值和方差的方法。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── welford_update
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── welford_update.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  welford_update单算子，对输入tensor做WelfordUpdate计算。Welford是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。WelfordUpdate接口为Welford算法的前处理。
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> welford_update </td></tr>

  <tr><td rowspan="5" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">srcGm</td><td align="center">1*64</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">inMeanGm</td><td align="center">1*64</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">inVarGm</td><td align="center">1*64</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="3" align="center">算子输出</td></tr>
  <tr><td align="center">outMeanGm</td><td align="center">1*64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outVarGm</td><td align="center">1*64</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">welford_update_custom</td></tr>
  </table>

- 算子实现：  
  本样例实现了welford_update算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用WelfordUpdate高阶API接口完成welford_update计算，得到最终结果，再搬出到外部存储上。

    welford_update算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm、inMeanGm、inVarGm存储在srcLocal、inMeanLocal、inVarLocal中，Compute任务负责对srcLocal、inMeanLocal、inVarLocal执行welford_update计算，计算结果存储在outMeanLocal、outVarLocal中，CopyOut任务负责将输出数据从outMeanLocal、outVarLocal搬运至Global Memory上的输出Tensor outMeanGm、outVarGm。

  - Tiling实现

    welford_update算子的tiling实现流程如下：首先获取welford_update接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入长度dataLength确定所需tiling参数。

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
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```