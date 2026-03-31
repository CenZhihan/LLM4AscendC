# Matmul实现ColumnMajor单算子直调样例

## 概述

本样例介绍了A、B、C矩阵为COLUMN_MAJOR（列优先）格式排布的矩阵乘的实现方式。不同于ND（行优先）格式排布的矩阵乘计算，对于COLUMN_MAJOR（列优先）格式排布的矩阵，Matmul API支持将矩阵设置为COLUMN_MAJOR格式。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_column_major
│   └── scripts
│       ├── gen_data.py             // 输入数据和真值数据生成脚本文件
│       └── verify_result.py        // 真值对比文件
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── matmul_column_major.asc     // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能： 
  MatmulColumnMajorCustom算子调用Matmul API计算时，将列方向上的元素在内存连续的A、B、C矩阵的Format参数设置为CubeFormat::COLUMN_MAJOR，实现列优先排布的矩阵乘法。算子实现了对输入的A、B矩阵做矩阵乘、加bias偏置的功能。

- 算子规格：  

    本样例中，算子实现支持的shape为：M = 428, N = 479, K = 528。
    <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MatmulColumnMajor</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
    <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">bias</td><td align="center">1 * N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">COLUMN_MAJOR</td><td align="center">-</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmulColumnMajorCustom</td></tr>
    </table>

- 算子实现： 
  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[K, M]；B为右矩阵，形状为[N, K]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[N, M]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一列都采用该bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象：C矩阵的Format设置为COLUMN_MAJOR。
          ```
          AscendC::Matmul<
            AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, ATYPE>,
            AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, BType>,
            AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, CType>,
            AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>> matmulObj;
          ```
      - 初始化操作。
      - 设置左矩阵A、右矩阵B、Bias。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
      - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
      - 获取Tiling参数的流程如下：
        - 创建一个Tiling对象。
        - 设置A、B、C、Bias的参数类型信息，其中A、B、C矩阵的Format设置为COLUMN_MAJOR。
          ```
          cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
              matmul_tiling::DataType::DT_FLOAT16, isAtrans);
          cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
              matmul_tiling::DataType::DT_FLOAT16, isBtrans);
          cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
              matmul_tiling::DataType::DT_FLOAT);
          ```
        - 设置M、N、Ka、Kb形状信息等。
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
  mkdir -p build && cd build;    # 创建并进入build目录
  cmake ..;make -j;    # 编译工程
  python3 ../scripts/gen_data.py    # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```