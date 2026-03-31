# Matmul常量化算子样例
## 概述
本样例介绍了实现Tiling常量化的Matmul单算子。Tiling常量化即在编译期期间将部分或全部Tiling参数由变量转化为常数值，在算子执行时将使用常量化的Tiling参数。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构
```
├── matmul_constant
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_constant.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  MatmulConstantCustom单算子，对输入的A、B矩阵做矩阵乘和加bias偏置，同时使用常量化Tiling方式实现Kernel侧静态编译，在算子执行前推导出Tiling信息。通过实现Matmul Tiling常量化，减少算子中的Scalar操作，提升算子性能。Tiling常量化场景中，在Kernel侧使用SetSingleShape接口设置的SingleShape是运行时单核计算的最大形状，实际计算时shape应该小于等于该形状。

- 算子规格： 

  本样默认执行的算子shape为：M = 128, N = 30720, K = 64。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">1 * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_constant_custom</td></tr>
  </table>
- 算子实现

  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 具体步骤：
      - 配置常量化的MatmulShapeParams，获取自定义MatmulConfig。
        ```
        constexpr int32_t MAX_M = 10000; // custom matmul kernel support max value of M Dim shape
        constexpr int32_t MAX_N = 10000; // custom matmul kernel support max value of N Dim shape
        constexpr int32_t MAX_K = 10000; // custom matmul kernel support max value of K Dim shape
        constexpr int32_t BASE_M = 128;  // BASEM * BASE_K * sizeof(typeC) <=L0A size
        constexpr int32_t BASE_N = 256;  // BASEN * BASE_K * sizeof(typeB) <=L0B size
        constexpr int32_t BASE_K = 64;   // BASEM * BASE_N * sizeof(typeC) <=L0C size
        constexpr MatmulShapeParams shapeParams = { MAX_M,
                                                      MAX_N,
                                                      MAX_K,
                                                      BASE_M,
                                                      BASE_N,
                                                      BASE_K };
        constexpr MatmulConfig CUSTOM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
        ```
      - 通过GetMatmulApiTiling接口获取常量化Tiling信息。
        ```
        auto constantCFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
        ```
      - 使用自定义MatmulConfig模板创建Matmul对象。
        ```
        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>;
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;
        using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
        constexpr static auto CONSTANT_CFG = GetCustomConstantCFG<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
        ```
      - 初始化操作，传入常量化Tiling信息。
      - 设置左矩阵A、右矩阵B、Bias。
        ```
        matmulObj.SetTail(tailM, tailN, shapes.k);
        matmulObj.SetTensorA(aGlobal, false);
        matmulObj.SetTensorB(bGlobal, false);
        if (shapes.isBias) {
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
      - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数，由于常量化Tiling，在Tiling中只需要实现分核的操作即可。用户可以通过多核Tiling接口，获取最优的多核切分策略。其他Tiling信息在Kernel侧通过常量化推导，Kernel侧不再需要运行时Tiling信息。

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