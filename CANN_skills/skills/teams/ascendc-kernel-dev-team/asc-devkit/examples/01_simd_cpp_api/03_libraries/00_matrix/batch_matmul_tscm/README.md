# Batch Matmul算子多batch L1输入场景样例
## 概述
本样例介绍了调用Matmul高阶API实现左矩阵为L1输入的BatchMatmul单算子，即算子中自定义TSCM输入。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构
```
├── batch_matmul_tscm
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── batch_matmul_tscm.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

     BatchMatmul单算子，批量处理Matmul计算，每次Matmul计算对输入的A、B矩阵做矩阵乘和加bias偏置。该算子中A矩阵使用TSCM输入。

- 算子规格： 

  本样默认执行的算子shape为：M = 32, N = 256, K = 64, BatchNum = 3。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K * BatchNum</td><td align="center">float16</td><td align="center">NZ</td></tr>
  <tr><td align="center">b</td><td align="center">K * N * BatchNum</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N * BatchNum</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">batch_matmul_tscm_custom</td></tr>
  </table>
- 算子实现
  - 约束条件
    - 在BatchMatmul的输入矩阵位置为L1时，输入输出的Layout只支持NORMAL。
    - TSCM输入的矩阵必须能在L1 Buffer上全载，且L1 Buffer上的数据应当为NZ格式。

  - Kernel实现
    - 一次完成BatchNum个Matmul矩阵乘法的运算。单次MatMul的计算公式为：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象。
        创建Matmul对象时，将A矩阵MatmulType的Position设为TSCM，Format设为NZ。
          ```
          using A_TYPE = AscendC::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, half, false, LayoutMode::NORMAL>;
          using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NORMAL>;
          using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
          using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
          constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
          constexpr MatmulBatchParams batchParams = {false, BatchMode::BATCH_LESS_THAN_L1, false};
          constexpr MatmulConfig CFG_MM = GetMMConfig<configMode>(batchParams);
          AscendC::Matmul<A_TYPE， B_TYPE，C_TYPE， BIAS_TYPE， CFG_MM> matmulObj;
          ```
      - 多batch矩阵A从GM到L1的搬运，矩阵A为TSCM输入, 矩阵B和bias（如果有）从GM搬入。
          ```
          uint32_t BatchNum = 3;
          AscendC::TSCM<AscendC::TPosition::GM, 1> scm;
          pipe->InitBuffer(scm, 1, BatchNum * tiling.M * tiling.Ka * sizeof(AType));
          auto scmTensor = scm.AllocTensor<AType>();
          DataCopy(scmTensor, aGlobal, BatchNum * tiling.M * tiling.Ka);
          scm.EnQue(scmTensor);
          AscendC::LocalTensor<AType> scmLocal = scm.DeQue<AType>();

          matmulObj.SetTensorA(scmLocal);
          matmulObj.SetTensorB(bGlobal);
          if (tiling.isBias) {
            matmulObj.SetBias(biasGlobal);
          }
          ```
      - 完成多batch矩阵乘操作。
          ```
          matmulObj.IterateBatch(cGlobal, BatchNum, BatchNum, false);
          ```
      - 结束矩阵乘操作。

  - Tiling实现
    - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul Kernel计算时所需的 Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
    - 获取Tiling参数的流程如下：
      - 创建一个Tiling对象。
      - 根据输入输出Layout设置单核计算的A、B、C、Bias的参数类型信息，将A矩阵MatmulType的Position设为TSCM，Format设为NZ；M、N、Ka、Kb形状信息等。
          ```
          cubeTiling.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ,
              matmul_tiling::DataType::DT_FLOAT16, isAtrans);
          cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT16, isBtrans);
          cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT);
          cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT);
          cubeTiling.SetBatchNum(BatchNum);
          ```
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