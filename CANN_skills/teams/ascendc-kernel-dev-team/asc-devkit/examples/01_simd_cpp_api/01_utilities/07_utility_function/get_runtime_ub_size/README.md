# GetRuntimeUBSize样例

## 概述

样例介绍GetRuntimeUBSize使用方法，接口返回运行时变量，表示用户最大可以使用的Unified Buffer大小（单位为Byte）。该接口适用于SIMT和SIMD混合编程场景，此时SIMT场景会在Unified Buffer预留一部分空间用于Dcache，系统会额外预留一部分空间，剩余空间为用户可最大使用空间。例如Ascend 950PR/Ascend 950DT场景，SIMT编程申请32KB空间用于Dcache，系统预留8KB，Unified Buffer整体大小为256KB，此时调用接口返回216KB。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── get_runtime_ub_size
│   ├── CMakeLists.txt                       // 编译工程文件
│   └── get_runtime_ub_size.asc              // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  样例基于Abs取绝对值运算进行功能说明，计算公式：
  ```
  z = Abs(x)
  ```
- 样例规格：
  <table border="2" align="center">
  <caption>表1：样例输入输出规格</caption>
  <tr><td rowspan="3" align="center">样例输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[126976]</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[126976]</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_get_runtime_ub_size</td></tr>
  </table>

- 调用实现

  使用内核调用符<<<>>>调用核函数。

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
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
