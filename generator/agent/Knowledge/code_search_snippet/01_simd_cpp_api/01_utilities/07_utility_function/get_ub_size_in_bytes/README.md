# GetUBSizeInBytes样例

## 概述

本样例介绍GetUBSizeInBytes接口使用方法，接口返回结果为编译时常量，表示用户最大可以使用的Unified Buffer大小（单位为Byte），例如：Ascend 950PR/Ascend 950DT场景，系统预留8KB，Unified Buffer总共为256KB，接口返回248KB，为用户能使用的上限。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── get_ub_size_in_bytes
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── get_ub_size_in_bytes.asc              // Ascend C样例实现 & 调用样例
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
  <tr><td align="center">x</td><td align="center">[16384]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">样例输出</td></tr>
  <tr><td align="center">z</td><td align="center">[16384]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">get_ub_size_in_bytes_custom</td></tr>
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
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
