# Erfc样例

## 概述

本样例演示了基于Erfc高阶API的算子实现。样例返回输入x的互补误差函数结果，积分区间为x到无穷大。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── erfc
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── erfc.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  返回输入x的互补误差函数结果，积分区间为x到无穷大。原始的理论计算公式如下：  
  $$
  dstTensor_i = Erfc(srcTensor_i)
  $$
  $$
  Erfc(x) = 1 - Erf(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{x}^{\infty} e^{-t^2} dt
  $$
  由于Erfc函数没有初等函数表达方式，一般通过函数逼近的方式计算，近似计算公式如下所示：  
  $$
  Erfc(x) = \exp(-x_a^2) * \frac{R(z)}{S(z)} * \frac{x}{x_a} + (1 - \frac{x}{x_a})
  $$
  $$
  x_a = |x| + \min_{fp32}
  $$
  $$
  z = \min(x_a, 10)
  $$

  其中，  
  R(z) = (((((((z * R0 + R1) * z + R2) * z + R3) * z + R4) * z + R5) * z + R6) * z + R7) * z + R8是关于z的8次多项式；  
  S(z) = ((((z + S1) * z + S2) * z + S3) * z + S4) * z + S5是关于z的4次多项式。

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> erfc </td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">4096</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center">4096</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">erfc_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src[4096]，输出dst[4096]的erfc_custom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Erfc高阶API接口完成Erfc计算，得到最终结果，再搬出到外部存储上。

    erfc_custom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行Erfc计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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