# LayerNormGradBeta样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用LayerNormGradBeta高阶API实现layernormgradbeta单算子，LayerNormGradBeta是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同，从而加速训练过程和提高模型的泛化能力（有效减少梯度消失和梯度爆炸问题）。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── layernormgradbeta
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── layernormgradbeta.asc   // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  layernormgradbeta单算子，LayerNormGradBeta是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同,基本思想是对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。具体来说，对于输入特征x，LayerNormGradBeta的计算过程可以表示为：  
  对输入特征x，在batch维度上计算均值μ和方差σ：  
  $$
  \mu_i = \frac{1}{B} \sum_{j = 1}^{B} x_{j,i} \quad \delta_i^2 = \frac{1}{B} \sum_{j = 1}^{B} (x_{j,i} - \mu_i)^2
  $$

  对于每个特征i，对输入特征x进行归一化：  
  $$
  \hat{x_{j,i}} = \frac{x_{j,i} - \mu_i}{\sqrt{\delta_i^2+\varepsilon}}
  $$

  对归一化后的特征进行缩放和平移：  
  $$
  y_{j,i}=\hat{\gamma}_{i} x_{j,i}+\beta_i = BN_{\gamma\beta}(x_{j,i})
  $$

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> layernormgradbeta </td></tr>

  <tr><td rowspan="4" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputDy</td><td align="center">4 * 8 * 512</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">resForGamma</td><td align="center">4 * 8 * 512</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="3" align="center">算子输出</td></tr>
  <tr><td align="center">outputPdGamma</td><td align="center">512</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputPdBeta</td><td align="center">512</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">layernormgradbeta_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape(inputDy[4 * 8 * 512]、resForGamma[4 * 8 * 512]， outputPdGamma[512]、 outputPdBeta[512])的layernormgradbeta_custom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用LayerNormGradBeta高阶API接口完成layernormgradbeta计算，得到最终结果，再搬出到外部存储上。

    layernormgradbeta算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor inputDyGm、resForGammaGm搬运至LocalMemory，分别存储在inputDyLocal、resForGammaLocal中，Compute任务负责对inputDyLocal、resForGammaLocal执行layernormgradbeta计算，计算结果存储在outputPdGammaLocal、outputPdBetaLocal中，CopyOut任务负责将输出数据从outputPdGammaLocal、outputPdBetaLocal搬运至Global Memory上的输出Tensor outputPdGammaGm、outputPdBetaGm中。

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