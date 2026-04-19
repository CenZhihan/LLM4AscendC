# SinCos样例

## 概述

本样例演示了基于SinCos高阶API的算子实现。样例按元素进行正弦计算和余弦计算，分别获得正弦和余弦的结果。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── sincos
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── sincos.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  按元素进行正弦计算和余弦计算，分别获得正弦和余弦的结果。

  计算公式如下：
  $$
  dst0_i = Sin(src_i)
  $$
  $$
  dst1_i = Cos(src_i)
  $$

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> sincos </td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="3" align="center">算子输出</td></tr>
  <tr><td align="center">dst0</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">dst1</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sincos_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src[1024]、src[1024]，输出dst0[1024]、dst1[1024]的sincos_custom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用SinCos高阶API接口完成SinCos计算，得到最终结果，再搬出到外部存储上。

    sincos_custom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行SinCos计算，计算结果存储在dst0Local、dst1Local中，CopyOut任务负责将输出数据从dst0Local、dst1Local搬运至Global Memory上的输出Tensor dst0Gm、dst1Gm。

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