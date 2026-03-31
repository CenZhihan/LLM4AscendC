# Reduce样例

## 概述

本样例介绍了调用WholeReduceSum和BlockReduceSum高阶API实现reduce算子，实现了连续内存上数据元素的累加，返回累加结果的功能。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── reduce
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── reduce.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  ReduceCustom算子实现了连续内存上数据元素的累加，返回累加结果的功能。对应的数学表达式为：

  ```
  z = sum(x)
  ```

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> reduce </td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">4096</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">y</td><td align="center">32</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入x[4096]，输出y[32]的ReduceCustom算子。x长度的取值范围为(0,4096]float；(0,16384]half。

    样例将一段连续的输入做累加，得到这段连续buffer内元素的和。

      - 1、在小于256B时，ComputeKey = 1，采用WholeReduceSum一次性可以得到结果。

      - 2、长度在float输入(256B,2KB]，或者half输入(256B,4KB]时，ComputeKey = 2。由于同样长度的输入，BlockReduceSum比WholeReduceSum的执行速度更快，所以采用一条BlockReduceSum加一条WholeReduceSum的做法，得到更高的计算效率。

      - 3、长度在float输入(2KB,16KB]，或者half输入(4KB,32KB]时，ComputeKey = 3。由于一条WholeReduceSum的累加效率比使用两条BlockReduceSum的累加效率更高。所以采用两条WholeReduceSum（而不是两条BlockReduceSum+一条WholeReduceSum），得到这段buffer的累加和。

      - 4、长度在float输入为10000时，ComputeKey = 4，对应WholeReduceSumImpl中的处理方法，在Counter模式下，采用WholeReduceSum指令，循环处理二维数据中的每一行，得到每一行的归约运行结果。

      - 5、长度在float输入为20000时，ComputeKey = 5，对应BinaryReduceSumImpl中的处理方法，在Counter模式下，先将运算数据一分为二，使用Add指令将两部分数据相加，循环往复，最后一条WholeReduceSum指令得到归约的运行结果。此种操作方式，相比较WholeReduceSum单指令操作的方式，在数据量较大，循环次数较多的场景下，性能更优。

      注意代码中使用了Counter模式。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Reduce高阶API接口完成reduce计算，得到最终结果，再搬出到外部存储上。

    ReduceCustom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm存储在xLocal中，Compute任务对xLocal执行reduce计算，计算的方式通过ComputeKey参数决定，ComputeKey由输入的长度决定，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm。

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