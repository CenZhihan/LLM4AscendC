# GroupBarrier样例

## 概述

本样例实现了两组存在依赖关系的AIV之间的正确同步，A组AIV计算完成后，B组AIV依赖该A组AIV的计算结果进行后续的计算，称A组为Arrive组，B组为Wait组。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── group_barrier
│   ├── CMakeLists.txt         // 编译工程文件
│   └── group_barrier.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  GroupBarrier算子实现了两组存在依赖关系的AIV之间的正确同步，A组AIV计算完成后，B组AIV依赖该A组AIV的计算结果进行后续的计算，称A组为Arrive组，B组为Wait组。本算子不进行输入输出计算，仅通过Arrive组写完指定数值之后，Wait组读取该数值，printf打印出正确的数值完成验证。

- 算子规格：  
  <table>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">barGm</td><td align="center">3072</td><td align="center">uint8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8</td><td align="center">int32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_group_barrier</td></tr>  
  </table>

- 算子实现：

  - Kernel实现  
    GroupBarrier算子启用8个AIV核，其中2个AIV核作为Arrive组开启原子累加，将指定数值写入Global Memory中，并调用Arrive指令；其余6个AIV核首先调用Wait指令等待Arrive组完成写入，然后读出Global Memory并将结果通过printf打印出来。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。

- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  [Block (0/6)]: OUTPUT = 24
  [Block (1/6)]: OUTPUT = 24
  [Block (2/6)]: OUTPUT = 24
  [Block (3/6)]: OUTPUT = 24
  [Block (4/6)]: OUTPUT = 24
  [Block (5/6)]: OUTPUT = 24
  ```
