# LayerNorm样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用LayerNorm高阶API实现layernorm单算子，对输入tensor按行做Layernorm计算。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── layernorm
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── layernorm.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  layernorm单算子，对输入tensor按行做Layernorm计算。
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> layernorm </td></tr>

  <tr><td rowspan="5" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputXGm</td><td align="center">2 * 32 * 16</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gammaGm</td><td align="center">16</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">betaGm</td><td align="center">16</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="4" align="center">算子输出</td></tr>
  <tr><td align="center">outputGm</td><td align="center">2 * 32 * 16</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputMeanGm</td><td align="center">2 * 32</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputVarianceGm</td><td align="center">2 * 32</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">layernorm_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape(x[2, 32, 16], gamma[16], beta[16])的layernorm算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用LayerNorm高阶API接口完成layernorm计算，得到最终结果，再搬出到外部存储上。

    layernorm算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor inputXGm、gammaGm、betaGm Memory搬运至LocalMemory，分别存储在inputXLocal、gammaLocal、betaLocal中，Compute任务负责对inputXLocal、gammaLocal、betaLocal执行layernorm计算，计算结果存储在outputLocal、meanLocal、varianceLocal中，CopyOut任务负责将输出数据从outputLocal、meanLocal、varianceLocal搬运至Global Memory上的输出Tensor outputGm、outputMeanGm、outputVarianceGm中。

  - Tiling实现

    layernorm算子的tiling实现流程如下：首先获取LayerNorm接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取LayerNorm kernel侧接口所需tiling参数。

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