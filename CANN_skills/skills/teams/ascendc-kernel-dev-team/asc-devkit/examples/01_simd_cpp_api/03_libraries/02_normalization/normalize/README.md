# Normalize样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用Normalize高阶API实现normalize单算子，LayerNorm中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和y。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── normalize
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── normalize.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  normalize单算子，对输入的均值Normalize和方差Variance做Normalize计算，得到Rstd和最后的归一化结果。
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> normalize </td></tr>

  <tr><td rowspan="7" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputX</td><td align="center">8*64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inputMean</td><td align="center">8</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inputVariance</td><td align="center">8</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gamma</td><td align="center">64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">64</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="3" align="center">算子输出</td></tr>
  <tr><td align="center">outputRstd</td><td align="center">8</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">output</td><td align="center">8*64</td><td align="center">float</td><td align="center">ND</td></tr>


  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">normalize_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape(inputX[8, 64]、inputMean[8]、inputVariance[8]、gamma[64]、beta[64]、outputRstd[8]、output[8, 64])的normalize算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Normalize高阶API接口完成normalize计算，得到最终结果，再搬出到外部存储上。

    normalize算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor inputXGm、inputMeanGm、inputVarGm、gammaGm、betaGm存储在inputXLocal、inMeanLocal、inVarLocal、inGammaLocal、inBetaLocal中，Compute任务负责对inputXLocal、inMeanLocal、inVarLocal、inGammaLocal、inBetaLocal执行normalize计算，计算结果存储在outLocal、outRstdLocal中，CopyOut任务负责将输出数据从outLocal、outRstdLocal搬运至Global Memory上的输出Tensor outGm、outRstdGm。

  - Tiling实现

    normalize算子的tiling实现流程如下：首先获取normalize接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入长度dataLength确定所需tiling参数。

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