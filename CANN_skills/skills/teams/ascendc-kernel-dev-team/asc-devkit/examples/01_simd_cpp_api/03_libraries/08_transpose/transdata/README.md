# TransData样例

## 概述

本样例演示了基于TransData高阶API实现的算子实现。样例将输入数据的排布格式转换为目标排布格式。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── transdata
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── transdata.asc           // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  将输入数据的排布格式转换为目标排布格式。  
  本样例支持的数据格式转换场景包括以下四种，样例中通过mode参数控制：    
  - 场景1：NCDHW -> NDC1HWC0， mode = 3
  - 场景2：NDC1HWC0 -> NCDHW， mode = 4
  - 场景3：NCDHW -> FRACTAL_Z_3D，  mode = 1
  - 场景4：FRACTAL_Z_3D -> NCDHW，  mode = 2

  除维度顺序变换外，其中涉及到C轴和N轴的拆分，具体转换方式为，C轴拆分为C1轴、C0轴，N轴拆分为N1轴、N0轴。  
  对于位宽为16的数据类型的数据，C0和N0固定为16，C1和N1的计算公式如下：  
  $$ C1 = (C + C0 - 1) / C0 $$
  $$ N1 = (N + N0 - 1) / N0 $$

- 算子规格：  
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center"> transdata </td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">16 * 16 * 1 * 3 * 5</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">dst</td><td align="center">16 * 16 * 1 * 3 * 5</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">transdata_custom</td></tr>
  </table>

- 算子实现：  
  本样例中实现的是固定shape为输入src[16, 16, 1, 3, 5]，输出dst[16, 16, 1, 3, 5]的transdata_custom算子。参数mode = 1，即数据格式转换场景为NCDHW -> FRACTAL_Z_3D。

  - Kernel实现

    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用TransData高阶API接口完成TransData计算，得到最终结果，再搬出到外部存储上。

    transdata_custom算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行TransData计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

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