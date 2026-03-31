# Matmul No IBShareAB特性样例

## 概述

本样例介绍了调用Matmul高阶API实现不开启IBShare功能的单算子。IBShare的功能是复用L1 Buffer上相同的A矩阵或者B矩阵数据，减少数据搬运开销。本样例为A矩阵和B矩阵同时不复用场景。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 推理系列产品AI Core

## 目录结构介绍

```
├── matmul_no_ibshareAB
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_no_ibshareAB.asc            // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能： 

  算子名中ABshare的含义为，Matmul计算的A矩阵与B矩阵同时使能类型信息MatmulType的参数IBShare；当A矩阵和B矩阵同时使能IBShare时，表示L1 Buffer上的A矩阵和B矩阵同时复用。  
  MatmulNoABshare算子的A矩阵与B矩阵均未使能IBSHARE，数据按照K列进行切分计算。结合样例[matmul_ibshareAB](../matmul_ibshareAB)，对比两个算子的运行时间，计算matmul_ibshareAB算子的性能提升百分比。

- 算子规格： 

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">128 * 384</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">384 * 256</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">128 * 256</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_noABshare_custom</td></tr>
  </table>

- 算子实现： 

  - Kernel实现  
    matmul_no_ibshareAB算子的数学表达式为：
    ```
    C = A * B
    ```
    - A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
    - C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
    其中A的形状为[128, 384]，B的形状为[384, 256]，C的形状为[128, 256]。

    matmul_no_ibshareAB算子代码数据处理说明图示(A矩阵和B矩阵不切分处理)：  
    ![alt text](./pictures/matmul_noABshare.png)


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
  python3 ../scripts/verify_result.py output/output_noABshare.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```