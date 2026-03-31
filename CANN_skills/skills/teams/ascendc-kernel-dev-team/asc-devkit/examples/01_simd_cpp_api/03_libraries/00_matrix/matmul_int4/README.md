# Matmul算子int4数据类型输入直调样例
## 概述
本样例介绍Matmul API实现int4数据类型输入，int32数据类型输出的单算子。
## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_int4
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_int4.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能：

  Matmul算子调用Matmul API计算时，通过Kernel侧设置左矩阵A、右矩阵B的数据类型参数为int4b_t，Host侧调用SetAType、SetBType接口设置左矩阵A、右矩阵B的数据类型为DT_INT4，实现int4数据类型输入的单算子。

- 算子规格： 

  本样例默认执行的算子shape为：M = 256, N = 7680, K = 128。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">int4b_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">int4b_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_int4_custom</td></tr>
  </table>
- 算子实现： 
  - 约束条件
    - 左矩阵A、右矩阵B的数据类型必须同时为int4，对应输出C矩阵的数据类型只支持int32、half。
    - 左矩阵A为int4数据类型时，不支持转置。
    - int4数据类型输入的Matmul对象仅支持Norm、MDL、IBShare模板。

  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象。
        ```
        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AscendC::int4b_t>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AscendC::int4b_t>;
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
        using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
        ```
      - 初始化操作。
      - 设置左矩阵A、右矩阵B。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。
  
  - Tiling实现
    - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
    - 获取Tiling参数的流程如下：
      - 创建一个Tiling对象。
      - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
        ```
        TCubeTiling tilingData;
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
        cubeTiling.SetDim(numBlocks);
        cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_INT4, isAtrans);
        cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_INT4, isBtrans);
        cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_INT32);
        cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_INT32);
        ...
        ```
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