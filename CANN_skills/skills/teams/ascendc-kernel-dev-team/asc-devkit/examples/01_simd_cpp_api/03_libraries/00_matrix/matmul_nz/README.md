# Matmul算子NZ格式直调样例

## 概述
本样例介绍Matmul API输入矩阵内轴非256B对齐的场景下，在AIV核上使用DataCopyPad实现ND转换NZ格式的单算子。能够避免随路非对齐搬移时效率较低，从而提升算子性能。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── matmul_nz
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── nd2nz_utils.h           // 数据ND to NZ格式转换函数
│   └── matmul_nz.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能： 

  Matmul算子对输入的A、B矩阵做矩阵乘和加bias偏置。如果A矩阵或者B矩阵的内轴非256B对齐（后文简称非对齐），可以在AIV核上使用DataCopyPad等指令，实现ND格式到NZ格式的转换（后文简称ND2NZ），解决随路非对齐搬移效率降低的问题。通常非对齐的矩阵数据量较大时，使用这种转换方式后，再将对齐的NZ数据搬入L1 Buffer的算子有较大的性能提升。
  
- 算子规格： 

  本样例默认执行的算子shape为：M = 1024, N = 4095, K = 1024。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_nz_custom</td></tr>
  </table>

- 算子实现： 
  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象。  
        创建Matmul对象时，定义内轴非256B对齐的B矩阵的Format为NZ格式。
        ```
        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, BType>; // Use Nz format
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
        using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
        ```
      - 利用AIV核的Vector计算单元实现ND2NZ格式转换。如下代码中MatrixBtoNZ为将B矩阵进行ND2NZ格式转换的函数。
        ```
        // Vector ND2NZ
        if ASCEND_IS_AIV {
            pipe->InitBuffer(ubBuf, TOTAL_UB_SIZE);
            MatrixBtoNZ<typename B_TYPE::T>(tempGM, bGMNZ, tiling, IS_TRANS_B, ubBuf, tiling.baseK,
                tiling.baseN); // Vector侧实现的ND2NZ函数
            SyncAll();
            // CV SYNC
            NotifyEvent<PIPE_MTE3>(4);
            return;
        }
        if ASCEND_IS_AIC {
            WaitEvent(4); // 等待AIV核完成ND2NZ格式转换
        }
        ```
      - 设置左矩阵A、右矩阵B、Bias。
        ```
        matmulObj.SetTensorA(aGlobal, IS_TRANS_A);
        matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
        if (IS_BIAS) {
            matmulObj.SetBias(biasGlobal);
        }
        ```
      - 完成矩阵乘操作。
        ```
        matmulObj.IterateAll(cGlobal);
        ```
      - 结束矩阵乘操作。
        ```
        matmulObj.End();
        ```

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