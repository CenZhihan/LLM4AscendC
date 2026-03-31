# Matmul B8直调样例

## 概述

本样例介绍了调用Matmul高阶API实现A、B矩阵为hifloat8、fp8_e4m3fn、fp8_e5m2数据类型输入，并使用MDL模板的Matmul单算子。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_b8
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_b8.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能： 

  MatmulB8Custom算子，对输入的A、B矩阵做矩阵乘和加bias偏置，实现hifloat8、fp8_e4m3fn、fp8_e5m2数据类型输入的Matmul算子。其中，输入数据类型为hifloat8时，A、B数据类型必须一致。

- 算子规格： 

  本样例中，算子实现支持的shape为：M = 428, N = 479, K = 158。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M*K</td><td align="center">hifloat8、fp8_e4m3fn、fp8_e5m2</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">K*N</td><td align="center">hifloat8、fp8_e4m3fn、fp8_e5m2</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M*N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmulB8Custom</td></tr>
  </table>
- 算子实现： 

  - Kernel实现
    - 计算逻辑是：Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。MatMul的计算公式为：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 实现MatMul矩阵乘运算的具体步骤如下：
      - 创建Matmul对象。
      - 初始化操作。
      - 设置左矩阵A、右矩阵B、Bias。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
      - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
      - 获取Tiling参数的流程如下：
        - 创建一个Tiling对象。
        - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
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
  - MODE ： 设置A、B的数据类型组合
    - 0 : 对应A hifloat8   B hifloat8场景
    - 1 : 对应A fp8_e4m3fn B fp8_e4m3fn场景
    - 2 : 对应A fp8_e5m2   B fp8_e5m2场景
    - 3 : 对应A fp8_e4m3fn B fp8_e5m2场景
    - 4 : 对应A fp8_e5m2   B fp8_e4m3fn场景
  ```bash
  mkdir -p build && cd build;        # 创建并进入build目录
  cmake .. -DDT_MODE=0;make -j;      # 编译工程, 0可选01234， 例如cmake .. -DDT_MODE=1;make -j;
  python3 ../scripts/gen_data.py  0  # 生成测试输入数据, 0可选01234， 例如python3 ../scripts/gen_data.py  1
  ./demo                             # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```