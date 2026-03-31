# AdjustSoftmaxres样例

## 概述

本样例介绍了调用AdjustSoftMaxRes高阶API实现AdjustSoftMaxRes算子，并按照核函数直调的方式分别给出了对应的端到端实现。本样例以直调的方式调用算子核函数。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── adjustsoftmaxres
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── adjustsoftmaxres.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  AdjustSoftMaxResCustom单算子，用于对SoftMax相关计算结果做后处理。当输入的max中存在指定的值，调整x中对应位置的数据为自定义的值。
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> adjustsoftmaxres </td></tr>

  <tr><td rowspan="4" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">32*32</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">max</td><td align="center">32*8</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">y</td><td align="center">32*32</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">adjust_softmax_res_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入x[32, 32]，输出y[32, 32]的AdjustSoftMaxResCustom算子。算子中需要判断的max中的值为0xFF7FFFFF，需要向y的结果数据中填充的值为0，将这两个值分别定义为两个全局常量，用于算子实现。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用AdjustSoftMax高阶API接口完成计算，得到最终结果，再搬出到外部存储上。

    AdjustSoftMaxRes算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGlobal和maxLocal分别存储在srcLocal和maxLocal中，Compute任务负责对maxLocal中的数据进行判断，并更改srcLocal数据。计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGlobal。

  - Tiling实现

    根据输入Tensor的长度和宽度确定所需tiling参数height、width、srcSize、dstSize。

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