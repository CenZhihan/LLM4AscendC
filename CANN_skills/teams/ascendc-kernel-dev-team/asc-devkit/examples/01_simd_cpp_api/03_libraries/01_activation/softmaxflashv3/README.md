# SoftmaxFlashV3样例

## 概述

本样例基于Kernel直调算子工程，介绍了调用SoftmaxFlashV3高阶api实现softmaxflashv3单算子，SoftmaxFlash增强版本，对应Softmax PASA算法。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── softmaxflashv3
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── softmaxflashv3.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  将输入tensor[m0, m1, ..., mt, n]（t大于或等于0）的非尾轴长度m0, m1, ..., mt相乘的结果看作m，则输入tensor的shape看作[m, n]。对输入tensor x的尾轴进行切分，分块个数为splitMeanCnt，切分后的tensor为x_cnti。按如下公式进行计算，其中x、inmax、insum、inmean为输入，M、S、E、A均为输出。  
  update为false：  
  $$
  A_1 = \text{rowmean}(x_{cnt})_i, i \in [0, \text{splitMeanCnt}]\\
  A_2 = \text{rowmean}(x_i), i \in [0, n]\\
  x_i = x_i - (A_2 - A_1) * (\alpha / (1 - \alpha))\\  
  A = A_2\\ 
  M_1 = \text{rowmax}(x_i), i \in [0, n]\\ 
  M = M_1\\
  M_2 = M\\ 
  \text{SoftmaxFlashV3}(z_i) = \exp(x_i - M_2), i \in [0, n]\\  
  S = \sum_{i}^{n} \exp(x_i - M_2)\\
  $$ 
  update为true：  
  $$
  A_1 = \text{rowmean}(x_{cnt})_i, i \in [0, \text{splitMeanCnt}]\\
  A_2 = \text{rowmean}(x_i), i \in [0, n]\\
  x_i = x_i - (A_2 - A_1) * (\alpha / (1 - \alpha))\\
  A = (A_2 + \text{inmean} * (\text{loopCnt} - 1)) / \text{loopCnt}\\
  M_1 = \text{rowmax}(x_i), i \in [0, n]\\
  C = (A_2 - A) * (\alpha / (1 - \alpha))\\
  P = (\text{inmean} - A) * (\alpha / (1 - \alpha))\\
  M = \max(C + M_1, P + \text{inmax})\\
  M_2 = M - C\\
  \text{SoftmaxFlashV3}(z_i) = \exp(x_i - M_2), i \in [0, n]\\
  E = \exp(\text{inmax}_i - M_2 + P)\\
  S = \sum_{i}^{n} \exp(x_i - M_2) + E * \text{insum}\\
  $$
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> softmaxflashv3 </td></tr>

  <tr><td rowspan="6" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center"> 8*2048 </td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">inMax</td><td align="center"> 8*8 </td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inSum</td><td align="center"> 8*8 </td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inMean</td><td align="center"> 8*8 </td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center"> 8*2048 </td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">softmaxflashv3_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src[8, 2048]，inMax[8, 8]，inSum[8, 8]，inMean[8, 8]， 输出dst[8, 2048]的softmaxflashv3算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用SoftmaxFlashV3高阶API接口完成softmaxflashv3计算，得到最终结果，再搬出到外部存储上。

    softmaxflashv3算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm、inMaxGm、inSumGm、inMeanGm搬运至Local Memory，存储在srcLocal、inMaxLocal、inSumLocal、inMeanLocal，Compute任务负责对srcLocal、inMaxLocal、inSumLocal、inMeanLocal执行softmaxflashv3计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据dstLocal搬运至Global Memory上的输出Tensor dstGm中。

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