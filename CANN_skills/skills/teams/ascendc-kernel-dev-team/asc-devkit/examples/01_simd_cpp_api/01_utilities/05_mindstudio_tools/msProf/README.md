# 基于MatmulLeakyRelu算子的msProf样例

## 概述

本样例基于MatmulLeakyRelu算子，演示了通过msProf采集上板和仿真的性能数据的方法。用户可根据输出的性能数据，快速定位算子的软、硬件性能瓶颈，提升算子性能的分析效率。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── matmul_leakyrelu
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── matmul_leakyrelu.asc    // Ascend C算子实现 & 调用样例
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
```

## 算子描述
  MatmulLeakyRelu算子通过将矩阵乘加（Matmul）与LeakyRelu激活函数计算实现，具体功能描述可参考[MatmulLeakyRelu算子详情](../../00_introduction/03_matmulleakyrelu/matmul_leakyrelu/README.md)章节。

## msProf工具介绍
msProf工具是单算子性能分析工具。包含msprof op和msprof op simulator两种使用方式。该工具协助用户定位算子内存、算子代码以及算子指令的异常，实现全方位的算子调优。当前支持基于不同运行模式（上板或仿真）和不同文件形式（可执行文件或算子二进制.o文件）进行性能数据的采集和自动解析。

- 上板性能采集
    
    通过上板性能采集，可以直接测定算子昇腾AI处理器上的运行时间。该方式适合在板环境中快速定位算子性能问题。
    
    基于可执行文件demo通过msprof op执行算子调优：
    ```
    msprof op ./demo
    ```

    - 性能数据说明

      命令完成后，会在默认目录下生成以“OPPROF_{timestamp}_XXX”命名的文件夹,性能数据文件夹结构示例如下：

      ```bash
      ├──dump                       # 原始的性能数据，用户无需关注
      ├──ArithmeticUtilization.csv  # cube/vector指令cycle占比
      ├──L2Cache.csv                # L2 Cache命中率，影响MTE2，建议合理规划数据搬运逻辑，增加命中率
      ├──Memory.csv                 # UB，L1和主存储器读写带宽速率
      ├──MemoryL0.csv               # L0A，L0B，和L0C读写带宽速率
      ├──MemoryUB.csv               # Vector和Scalar到UB的读写带宽速率
      ├──OpBasicInfo.csv            # 算子基础信息
      ├──PipeUtilization.csv        # 采集计算单元和搬运单元耗时和占比
      ├──ResourceConflictRatio.csv  # UB上的 bank group、bank conflict和资源冲突率在所有指令中的占比
      └──visualize_data.bin         # MindStudio Insight呈现文件
      ```
- 性能流水仿真
    
  通过性能流水仿真，可以在不实际部署到昇腾AI处理器上的情况下，模拟算子的运行性能，通过指令流水图、算子代码热点图、内存通路吞吐率波形图等可视化数据，以便进一步分析优化算子计算瓶颈。
    
    基于可执行文件demo通过msprof op simulator执行算子仿真调优：
    ```
    msprof op simulator --soc-version=Ascendxxxyy ./demo_sim
    ```
    - soc_version：昇腾AI处理器型号，如果无法确定具体的soc_version，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。
    - 仿真数据说明
    
      命令完成后，会在默认目录下生成以“OPPROF_{timestamp}_XXX”命名的文件夹, 仿真数据文件夹结构示例：

      ```bash
      ├──dump                    # 原始的性能数据，用户无需关注
      └──simulator               # 算子基础信息
        ├──core0.cubecore0      # 按照core*.veccore*或core*.cubecore*目录存放各核的仿真指令流水图文件
        ├──core0.veccore0
        ├──core0.veccore1
        ├──trace.json           # 仿真流水图和仿真热点函数等信息可视化呈现文件
        └──visualize_data.bin   # MindStudio Insight呈现文件

      ```

更多msProf工具使用方式请参考[MindStudio工具](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev)算子调优（msProf）中的内容。


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
  mkdir -p build && cd build;           # 创建并进入build目录
  cmake ..;make -j;                     # 编译工程
  python3 ../scripts/gen_data.py        # 生成测试输入数据
  msprof op ./demo                      # 基于可执行文件demo通过msprof op或msprof op simulator执行算子调优
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```