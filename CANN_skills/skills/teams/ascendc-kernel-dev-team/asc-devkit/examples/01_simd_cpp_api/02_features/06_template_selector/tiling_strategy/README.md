# 使用Tiling策略实现VectorAdd算子样例

## 概述

本样例实现了一个支持多种数据类型的向量加法算子，其核心在于采用精细的多核并行数据切分（Tiling）策略，通过17个参数动态管理任务分配，处理核间负载不均及尾核尾块场景，以实现高效能计算。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 08_tiling
│   ├── scripts
│   │   ├── gen_data.py          // 输入数据和真值数据生成脚本
│   │   └── verify_result.py     // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt           // 编译工程文件
│   ├── data_utils.h             // 数据读入写出函数
│   └── tiling_strategy.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
  ```
  z = x + y
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)
  </td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">tiling_strategy_custom</td></tr>
  </table>
- 算子实现：  
  Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先计算出不同核上计算的数据相对首地址的偏移并搬运进片上存储。如果涉及到Add算子不支持的数据类型，则需要申请额外的tmpBufeer，并在计算前使用Cast算子进行类型转换，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

  - Tiling实现  
    TilingData参数设计，TilingData参数本质上是和并行数据切分相关的参数。

    本样例尝试将输入数据均分在给定数目的核上计算，如不能均分，则计算整核和尾核的个数，整核和尾核分别处理不同数据量的数据。定义UB_BLOCK_NUM最为最大可用的UB数据块，对应UB的大小为UB_BLOCK_NUM * 32 Bytes。 并以UB_BLOCK_NUM为粒度进行tiling切分。算子中使用了17个tiling参数：
    - totalLength、dataType为输入数据的总长度和数据类型；
    - blockLength、tileNum、 tileLength、lastTileLength为核均分场景下每个核上的计算量、tiling切分个数以及整块和尾块长度；
    - former(tail)Num、former(tail)Length、former(tail)TileNum、former(tail)TileLength、former(tail)LastTileLength为核不均分的场景下整核和尾核相关的切分数据，与核均分场景下对应；
    - isEvenCore用来判断当前是否为核均分场景。

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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```