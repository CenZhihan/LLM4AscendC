# IsInf样例

## 概述

本样例演示了基于IsInf高阶API的算子实现。样例按元素判断输入的浮点数是否为 $\pm$ inf，输出结果为浮点数或布尔值。当输入为浮点类型时，对于 $\pm$ inf的输入数据，对应位置的结果为浮点类型的1，反之为0；当输入为布尔类型时，对于 $\pm$ inf的输入数据，对应位置的结果为true，反之为false。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── isinf
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── isinf.asc               // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  按元素判断输入的浮点数是否为 $\pm$ inf，输出结果为浮点数或布尔值。当输入为浮点类型时，对于 $\pm$ inf的输入数据，对应位置的结果为浮点类型的1，反之为0；当输入为布尔类型时，对于 $\pm$ inf的输入数据，对应位置的结果为true，反之为false。

  计算公式如下：
  $$
  dst_i = IsInf(src_i)
  $$
  当输入为浮点类型时:
  $$
  IsInf(src) = 
  \begin{cases}
  0.0, & src \neq \pm inf \\ 
  1.0, & src = \pm inf
  \end{cases}
  $$
  当输入为布尔类型时:
  $$
  IsInf(src) = 
  \begin{cases}
  false, & src \neq \pm inf \\ 
  true, & src = \pm inf
  \end{cases}
  $$

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> isinf </td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">isinf_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src[1024]，输出dst[1024]的isinf_custom算子。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用IsInf高阶API接口完成IsInf计算，得到最终结果，再搬出到外部存储上。

    isinf_custom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行IsInf计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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