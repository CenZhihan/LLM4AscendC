# Batch Matmul单算子直调样例

## 概述

本样例介绍了调用Matmul高阶API实现batchMatmul单算子。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── batch_matmul
│   └── scripts
│       ├── gen_data.py              // 输入数据和真值数据生成脚本文件
│       └── verify_result.py         // 真值对比文件
│   ├── CMakeLists.txt               // 编译工程文件
│   ├── data_utils.h                 // 数据读入写出函数
│   └── batch_matmul.asc             // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  batchMatmul单算子，对输入的A B矩阵做矩阵乘和加bias偏置。调用样例中实现的是固定shape为[M, N, K] = [192, 1536, 64], bias = [1536]的原始矩阵，A矩阵按照[B, S, N, G, D] = [2, 32, 1, 3, 64]输入, B矩阵按照[B, S, N, G, D] = [2, 256, 1, 3, 64]输入，bias矩阵按照[B, S, N, G, D] = [2, 1, 1, 3, 256]输入，C矩阵按照[B, S, N, G, D] = [2, 32, 1, 3, 256]输出，batchNum = 3的batchMatmul算子。

- 算子规格：  
  本样例默认执行的算子shape为：M = 192, N = 1536, K = 64。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="6" align="center">BatchMatmulCustom</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td><td align="center">layout</td></tr>
  <tr><td align="center">a</td><td align="center">[2, 32, 1, 3, 64]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td><td align="center">BSNGD</td></tr>
  <tr><td align="center">b</td><td align="center">[2, 256, 1, 3, 64]</td><td align="center">float16</td><td align="center">ND</td><td align="center">true</td><td align="center">BSNGD</td></tr>
  <tr><td align="center">bias</td><td align="center">[2, 1, 1, 3, 256]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">[2, 32, 1, 3, 256]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td><td align="center">BSNGD</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="6" align="center">batch_matmul_custom</td></tr>
  </table>
- 算子实现：  
  - Kernel实现
    - 一次完成batchNum个Matmul矩阵乘法的运算。单次MatMul的计算公式为：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[2, 32, 1, 3, 64]；B为右矩阵，形状为[2, 256, 1, 3, 64]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[2, 32, 1, 3, 256]。
      - Bias为矩阵乘偏置，形状为[2, 1, 1, 3, 256]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 实现Matmul矩阵乘运算的具体步骤如下：
      - 创建Matmul对象。
      - 初始化操作。
      - 设置左矩阵A 、右矩阵B、Bias。
      - 完成多batch矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
    - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的 Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
    - 获取Tiling参数的流程如下：
      - 创建一个Tiling对象。
      - 根据输入输出LayOut设置单核计算的A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
      - 调用SetALayout、SetBLayout、SetCLayout、SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数。
      - 调用GetTiling接口，获取Tiling信息。

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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  test pass!
  ```
