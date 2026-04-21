# PhiloxRandom样例

## 概述

本样例介绍了基于Ascend C异构混合编程模型演示PhiloxRandom融合算子的核函数直调实现，基于Philox随机数生成算法，给定随机数种，生成若干的随机数。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── philoxrandom
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── philoxrandom.asc        // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  根据最后一轴的方向对各元素求平均值。  
  如果输入是向量，则在向量中对各元素相加求平均；如果输入是矩阵，则沿最后一个维度对元素求平均。本接口最多支持输入为二维数据，不支持更高维度的输入。
- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> philoxrandom </td></tr>

  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center">1280</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">philoxrandom_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输出dst[1280]的PhiloxRandomCustom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，根据输入的参数使用PhiloxRandom高阶API接口完成philoxrandom计算，得到最终结果，再搬出到外部存储上。

    PhiloxRandomCustom算子的实现流程分为2个基本任务：Compute，CopyOut。Compute任务根据参数执行philoxrandom计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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