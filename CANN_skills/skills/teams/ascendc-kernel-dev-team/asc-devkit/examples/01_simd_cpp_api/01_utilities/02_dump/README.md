# MmadCustomDump样例

## 概述

本样例以矩阵乘计算为例，介绍核函数维测接口asc_dump和printf的使用方法，实现NPU侧函数张量数据的Dump和运算参数的输出。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 02_dump
│   ├── CMakeLists.txt         // 编译工程文件
│   ├── half.hpp               // 数据类型依赖文件
│   └── mmad_custom_dump.asc   // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  样例基于矩阵乘计算为背景，演示asc_dump系列接口（含asc_dump_gm、asc_dump_ubuf、asc_dump_cbuf和asc_dump_l1buf）在NPU侧样例核函数中的使用方法，通过调用上述接口实现不同物理位置上的张量数据可视化。

  此外，该系列接口兼容AscendC::DumpTensor接口。但后续开发中，建议优先使用asc_dump系列接口；若需Dump指定偏移位置的数据，由于asc_dump系列暂不支持该能力，可继续使用DumpAccChkPoint接口。

- 样例规格：
  - Mmad样例：

    矩阵乘规格：M = 16, N = 16, K = 16，详细信息如下表：
    <table>
    <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Mmad</td></tr>
    </tr>
    <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
    <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad_custom</td></tr>
    </table>

- 样例实现：

  1）.asc_dump系列接口的函数签名一样，仅函数名存在区分，这种区分用以表示要Dump的具体位置：
  ```cpp
  __aicore__ inline void asc_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dumpSize);
  __aicore__ inline void asc_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize);
  ```
  2）.在数据搬入阶段，通过asc_dump_gm接口Dump位于全局内存 (GM) 中的输入矩阵A、矩阵B与偏执矩阵Bias的原始数据；待数据加载至L0C并完成矩阵乘计算后，再通过asc_dump_cbuf接口Dump并输出L0C中的最终计算结果，完整呈现数据从GM输入到L0C计算完成的全流程状态。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake ..;make -j;                # 编译工程
  ./demo                           # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```