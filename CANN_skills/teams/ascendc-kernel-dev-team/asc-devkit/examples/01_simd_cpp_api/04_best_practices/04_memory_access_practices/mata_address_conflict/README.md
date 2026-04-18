# 同地址冲突算子直调样例
## 概述

本样例介绍了同地址冲突的影响以及两种解决方法，并提供核函数直调方法。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── 05_mata_address_conflict
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py               // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── data_utils.h                       // 数据读入写出函数
│   └── mata_address_conflict.asc          // Ascend C算子实现 & 调用样例
```

## 算子描述  
- 算子功能：  
  Adds算子实现了一个Tensor与标量值2.0相加，返回相加结果的功能。

  对应的数学表达式为：

  ```python
  z = x + 2.0
  ```

  - x：输入，形状为\[8192, 128]，数据类型为float；
  - z：输出，形状为\[8192, 128]，数据类型为float；

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Adds</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8192 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8192 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">adds_custom_v1/adds_custom_v2/adds_custom_v3</td></tr>
  </table>


- 算子实现：

  - kernel实现   
  
    计算逻辑是：本样例主要介绍数据搬运中的同地址冲突对搬运效率的影响，在Global Memory的数据访问中，数据访问请求(读/写)在AI 处理器内部会按照512 Bytes对齐进行地址转换，同一时刻如果多核的数据访问请求在转换后落在连续的512 Bytes范围内，出于数据一致性的要求，AI 处理器会对落入同一个512Bytes范围内的请求进行串行处理，导致搬运效率降低，即发生了同地址访问现象。

    本样例中共有3个实现版本：
    adds_custom_v1：基础实现版本，每个核的计算顺序一致，存在同地址冲突，带宽效率较差。
    adds_custom_v2：通过调整每个核的计算顺序，避免发生同地址冲突。
    adds_custom_v3：通过调整切分顺序，避免发生同地址冲突。

    当前算子执行机制保证用户kernel入参（包括workspace/tiling）的地址是512 Bytes对齐的，因此用户只需要根据地址的偏移量即可判断两个地址是否会落入连续的512 Bytes范围内。


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
  python3 ../scripts/verify_result.py output/output_z_1.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  python3 ../scripts/verify_result.py output/output_z_2.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  python3 ../scripts/verify_result.py output/output_z_3.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```