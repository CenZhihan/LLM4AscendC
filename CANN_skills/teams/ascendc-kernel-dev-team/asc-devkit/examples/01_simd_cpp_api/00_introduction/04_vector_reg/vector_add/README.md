# 基于RegBase编程add样例

## 概述
本样例基于RegBase编程范式实现Add计算，样例仅支持单核运行。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── vector_add
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── vector_add.asc                 // AscendC样例实现 & 调用样例
```

## 样例描述
- 样例功能：

  样例输入一个 [64, 64]的矩阵，沿着行方向做逐行累加操作，由于每一行依赖上一行计算的结果，在代码中插入了写-读同步。
- 样例规格：
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="3" align="center">AIV样例</td></tr>
  <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[64, 64]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[64, 64]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">vector_add</td></tr>
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
  mkdir -p build && cd build;                                               # 创建并进入build目录
  cmake ..;make -j;                                                         # 编译工程
  python3 ../scripts/gen_data.py                                            # 生成测试输入数据
  ./demo                                                                    # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```