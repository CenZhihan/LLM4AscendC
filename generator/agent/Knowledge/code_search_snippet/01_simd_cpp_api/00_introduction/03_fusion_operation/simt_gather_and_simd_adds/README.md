# SIMT与SIMD混合编程实现gather和adds计算

## 概述
本样例基于SIMT和SIMD混合编程模式实现gather和adds计算，以SIMT编程方式实现离散内存访问操作gather，以SIMD编程方式实现连续内存访问操作adds。


## 支持的产品
- Ascend 950PR/Ascend 950DT
## 目录结构
```
├── simt_gather_and_simd_adds
│   ├── CMakeLists.txt         # cmake编译文件
│   ├── gather_and_adds.asc    # Ascend C算子实现 & 调用样例
|   └── README.md
```

## 样例描述

- 样例功能：

  计算公式：

  ```
  output[i] = input[index[i]] + 1
  ```

- 样例规格：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">gather & adds</td></tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[100000]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">[8192]</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">output</td><td align="center">[8192]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gather_and_adds_kernel</td></tr>
  </table>

- 样例实现：

  Vector Core中SIMT单元和SIMD单元共享片上存储，可以使用片上存储完成SIMT和SIMD的混合编程。本例中样例输入index的shape为[8192]，可设置核数为8，每个核处理数据量为1024，设置线程数THREAD_COUNT为1024，每个线程处理1个数据元素，单个核只需调用1次simt_gather函数即可完成gather运算。

  > ⚠️ **注意** 当单核处理数据量大于设置的线程数时，需要切分数据到多个线程块，可使用asc_vf_call多次调用simt_gather函数启动多个线程块完成获取指定索引数据的操作。

  基于上述数据拆分，在simd_adds函数中，处理1024个数据元素的加1操作。

  > ⚠️ **注意** simd_adds中加1运算实际可以直接在simt_gather函数中快速实现，本例目的仅仅是通过一个简单用例展示SIMT和SIMD两种编程模式的混合编程方式，不是该样例最佳实践。

  gather & adds样例的实现流程主要分为3个步骤：simt_gather，simd_adds和DataCopy。

  （1）simt_gather从Global Memory输入中获取指定索引的数据。
  ```
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ...
  uint32_t gather_idx = index[idx];
  ...
  gather_output[threadIdx.x] = input[gather_idx];
  ```

  （2）simd_adds将Local Memory中数据做加1操作。调用Reg::LoadAlign将数据从Local Memory搬运到寄存器上，调用Reg::Adds完成加1运算并输出到目标寄存器，最后调用Reg::StoreAlign将数据从寄存器搬运到Local Memory。重复上述操作即可完成1024个数据元素的加1运算。
  ```
  for (uint16_t i = 0; i < repeat_times; i++) {
      AscendC::Reg::LoadAlign(src_reg0, input + i * one_repeat_size);
      AscendC::Reg::Adds(dst_reg0, src_reg0, ADDS_ADDEND, mask_reg);
      AscendC::Reg::StoreAlign(output + i * one_repeat_size, dst_reg0, mask_reg);
  }
  ```

  （3）DataCopy负责将输出数据从Local Memory搬运至Global Memory上。

- 调用实现：
  - CPU调测模式使用ICPU_RUN_KF CPU调测宏调用核函数。
  - NPU模式使用内核调用符<<<>>>调用核函数。

  应用程序通过ASCENDC_CPU_DEBUG宏区分代码逻辑运行于CPU模式还是NPU模式。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
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
  cmake ..; make -j;            # 编译工程
  ./demo                        # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```

## CPU Debug调测
CPU Debug功能支持对CPU执行过程中的运行状态进行调试，主要通过GDB工具实现。GDB调试支持设置断点、查看寄存器和内存状态、单步执行、查看调用栈等常用调试操作，并支持多线程程序的调试。

- 样例执行
  ```bash
  mkdir -p build && cd build;          # 创建并进入build目录
  cmake -DRUN_MODE=cpu ..; make -j;    # 编译工程
  ./demo                               # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```
  [Success] Case accuracy is verification passed.
  ```

- 进入GDB模式调试
  在上述指令中"./demo"前加入"gdb --args"，再次执行指令即可进入GDB模式。调试时需要选择子进程进行调试。
  ```bash
  mkdir -p build && cd build;
  cmake -DRUN_MODE=cpu ..; make -j;
  gdb --args ./demo
  ```
