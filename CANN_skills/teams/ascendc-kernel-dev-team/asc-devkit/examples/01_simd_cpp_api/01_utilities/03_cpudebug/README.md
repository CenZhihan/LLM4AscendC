# CPU Debug直调样例说明

## 概述

本样例介绍CPU调测能力，该调测模式将kernel代码在host侧编译执行，可通过GDB进行单步调试，GDB调试支持设置断点、查看寄存器和内存状态、单步执行、查看调用栈等常用调试操作，并支持多线程程序的调试。

## 支持的产品

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 目录结构介绍

```
├── 03_cpudebug
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── data_utils.h            // 数据读入写出函数
│   ├── CMakeLists.txt          // 编译工程文件
│   └── add.cpp                 // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  以Add计算为基础，介绍CPU调试能力，Add计算公式：
  ```
  z = x + y
  ```
- 样例规格：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

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

  ```bash
  # 选择芯片型号
  soc_version=${1:-soc_version}
  ```
  - soc_version：昇腾AI处理器型号，如果无法确定具体的soc_version，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。

- 样例执行
  ```bash
  mkdir -p build && cd build;
  cmake .. -Dsoc_version=${soc_version}; make -j
  python3 ../scripts/gen_data.py
  ./add
  python3 ../scripts/verify_result.py output.bin golden.bin
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```

- 进入gdb模式调试
  在上述指令中"./add"前加入"gdb --args"，再次执行指令即可进入gdb模式。
  ```bash
  mkdir -p build && cd build;
  cmake .. -Dsoc_version=${soc_version}; make -j
  python3 ../scripts/gen_data.py
  gdb --args ./add
  run                # 让程序在gdb模式正常执行直到结束
  quit               # 退出gdb模式
  python3 ../scripts/verify_result.py output.bin golden.bin
  ```