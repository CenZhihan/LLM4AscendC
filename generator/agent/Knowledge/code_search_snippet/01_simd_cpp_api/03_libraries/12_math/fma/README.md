# Fma样例

## 概述

本样例演示了基于Fma高阶API的算子实现。样例按元素计算两个输入相乘后与第三个输入相加的结果。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── fma
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── fma.asc                 // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  按元素计算两个输入相乘后与第三个输入相加的结果。

  计算公式如下：
  $$
  dst_i = Fma(src0_i, src1_i, src2_i)
  $$
  $$
  Fma(src0_i, src1_i, src2_i) = src0_i * src1_i + src2_i
  $$

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> fma </td></tr>

  <tr><td rowspan="5" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">128</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">128</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src2</td><td align="center">128</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center">128</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">fma_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src0[128]、src1[128]、src2[128]，输出dst[128]的fma_custom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Fma高阶API接口完成Fma计算，得到最终结果，再搬出到外部存储上。

    fma_custom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor src0Gm、src1Gm、src0Gm存储在srcLocal中，Compute任务负责对src0Local、src1Local、src2Local执行Fma计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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