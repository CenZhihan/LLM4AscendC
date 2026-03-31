# Matmul ub2l1 kfc样例

## 概述

本样例通过Matmul高阶API注册KFC，基于DataCopy实现UB(Unified Buffer)到L1(L1 Buffer)的数据搬运。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_ub2l1_kfc
│   ├── scripts
│   │   ├── gen_data.py                       // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                        // 编译工程文件
│   ├── data_utils.h                          // 数据读入写出函数
│   └── matmul_ub2l1_kfc.asc               // AscendC样例实现 & 调用样例
```

## 样例描述

- 样例功能：  
  通过Matmul高阶API注册KFC，将数据从UB(Unified Buffer)搬运到L1(L1 Buffer)，然后进行Mmad计算，底层是UB(Unified Buffer)->GM(Global Memory)->L1(L1 Buffer)的搬运。
- 样例规格：
  <table>
  <tr><td rowspan="3" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">z</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_ub2l1_kfc</td></tr>
  </table>
- 样例实现：  
  - 使用Matmul高阶API注册KFC组件。
  - 将数据从GM(Global Memory)搬运到UB(Unified Buffer)。
  - 将数据从UB(Unified Buffer)搬运到L1(L1 Buffer)，使用KFC进行通信。
  - 调用基础API LoadData将数据从L1(L1 Buffer)搬运到A2(L0A Buffer)与B2(L0B Buffer)。
  - 调用基础API Mmad进行矩阵乘计算。
  - 调用基础API Fixpipe将数据从CO1(L0C Buffer)搬运到GM(Global Memory)。
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