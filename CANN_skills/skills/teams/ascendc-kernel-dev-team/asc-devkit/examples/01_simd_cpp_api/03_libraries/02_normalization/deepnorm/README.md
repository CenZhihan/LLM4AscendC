# DeepNorm样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用DeepNorm高阶API实现deepnorm单算子，在深层神经网络训练过程中，执行层LayerNorm归一化时，可以使用DeepNorm进行替代，通过扩大残差连接来提高Transformer的稳定性。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── deepnorm
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── deepnorm.asc            // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  本样例实现了对shape大小为[B，S，H]的输入数据的DeepNorm归一化，其计算公式如下：

  DeepNorm(x) = LayerNorm(α * X + SubLayer(X))

  SubLayer(X)通常是指在DeepNorm模型中的一个子层（sub-layer），用于实现自注意力机制（self-attention mechanism）。本接口中会整体作为一个输入Tensor传入。  

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> deepnorm </td></tr>

  <tr><td rowspan="6" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputX</td><td align="center">4 * 16 * 64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inputGx</td><td align="center">4 * 16 * 64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gamma</td><td align="center">64</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="4" align="center">算子输出</td></tr>
  <tr><td align="center">output</td><td align="center">4 * 16 * 64</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputMean</td><td align="center">4 * 16</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputVariance</td><td align="center">4 * 16</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">deepnorm_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape(inputX[4, 16, 64]、inputGx[4, 16, 64]、beta[64]、gamma[64]， output[4, 16, 64]、 outputMean[4, 16]、 outputVariance[4, 16])的deepnorm算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用DeepNorm高阶API接口完成deepnorm计算，得到最终结果，再搬出到外部存储上。

    deepnorm算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor inputX_gm、inputGx_gm、gamma_gm、beta_gm Memory搬运至LocalMemory，分别存储在inputXLocal、inputGxLocal、gammaLocal、betaLocal中，Compute任务负责对inputXLocal、inputGxLocal、gammaLocal、betaLocal执行deepnorm计算，计算结果存储在outputLocal、outputMeanLocal、outputVarianceLocal中，CopyOut任务负责将输出数据从outputLocal、outputMeanLocal、outputVarianceLocal搬运至Global Memory上的输出Tensor output、outputMeanGm、outputVarianceGm中。

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