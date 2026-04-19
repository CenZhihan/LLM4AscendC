# Matmul实现ColumnMajor样例直调样例

## 概述

输入输出矩阵为COLUMN_MAJOR（列优先）格式排布的Matmul样例。不同于ND（行优先）格式排布的矩阵乘计算，对于COLUMN_MAJOR（列优先）格式排布的矩阵，Matmul API支持将矩阵设置为COLUMN_MAJOR格式。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_format_column_major
│   └── scripts
│       ├── gen_data.py             // 输入数据和真值数据生成脚本文件
│       └── verify_result.py        // 真值对比文件
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── matmul_format_column_major.asc     // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能： 
  MatmulColumnMajorCustom样例调用Matmul API计算时，将列方向上的元素在内存连续的A、B、C矩阵的Format参数设置为CubeFormat::COLUMN_MAJOR，实现列优先排布的矩阵乘法。样例实现了对输入的A、B矩阵做矩阵乘、加bias偏置的功能。

- 样例规格：  

    本样例中：M = 428, N = 479, K = 528。
    <table>
    <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="5" align="center">MatmulColumnMajor</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
    <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">COLUMN_MAJOR</td><td align="center">false</td></tr>
    <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">COLUMN_MAJOR</td><td align="center">-</td></tr>
    </tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmulColumnMajorCustom</td></tr>
    </table>

- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象：C矩阵的Format设置为COLUMN_MAJOR。
      ```cpp
      AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, ATYPE>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::COLUMN_MAJOR, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>> matmulObj;
      ```

  - Tiling关键步骤
    - 设置A、B、C、Bias的参数类型信息，其中A、B、C矩阵的Format设置为COLUMN_MAJOR。
      ```cpp
      cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
          matmul_tiling::DataType::DT_FLOAT16, isAtrans);
      cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
          matmul_tiling::DataType::DT_FLOAT16, isBtrans);
      cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::COLUMN_MAJOR,
          matmul_tiling::DataType::DT_FLOAT);
      ```

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