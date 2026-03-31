# Matmul M/N轴方向的流水并行算子样例
## 概述
本样例介绍了调用Matmul高阶API实现Matmul支持M/N轴方向的流水并行单算子。该功能的应用场景为输入矩阵的K很小，但M或N很大的场景，即singleCoreK<=baseK，但singleCoreM远大于baseM或singleCoreN远大于baseN。使能M/N方向流水并行功能可能会带来性能收益。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构
```
├── matmul_mndb
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_mndb.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  MatmulMNDBCustom单算子调用Matmul高阶API对输入的A，B矩阵做矩阵乘和加bias偏置。在定义Matmul对象时，将配置scheduleType等参数的MatmulConfig作为模板参数传入，使能M/N轴方向的流水并行功能。本样例以MDL模板且N方向流水并行为例，对于Norm模板或M方向流水并行功能也可以参考该样例的实现。

- 算子规格： 

  本样默认执行的算子shape为：M = 128, N = 7680, K = 16
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_mndb_custom</td></tr>
  </table>
- 算子实现
  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象。  
        创建Matmul对象时，自定义MatmulConfig参数，将其中的MatmulConfigMode设置成CONFIG_MDL，scheduleType参数设置为ScheduleType::OUTER_PRODUCT，iterateOrder参数设置为IterateOrder::ORDER_M，使能MDL模板的N方向流水并行功能，获得自定义地使用MDL模板的Matmul对象。
          ```
          constexpr static MatmulConfigMode configModeMDL = MatmulConfigMode::CONFIG_MDL;
          constexpr static MatmulFuncParams funcParamsOrderM{false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT, true, true};
          constexpr static MatmulConfig CFG_MDL_OUTER_PRODUCT_ORDER_M = GetMMConfig<configModeMDL>(funcParamsOrderM);

          using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE, false>;
          using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false>;
          using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
          using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
          AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL_OUTER_PRODUCT_ORDER_M> matmulObj;
          ```
      - 初始化操作。
      - 设置左矩阵A、右矩阵B、Bias。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
    - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul Kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
    - 获取Tiling参数的流程如下：
      - 创建一个Tiling对象。
      - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
      - 设置自定义MatmulConfig参数，将Kernel侧配置的参数如scheduleType等，同步到Tiling侧。
        ```
        matmul_tiling::MatmulConfigParams matmulConfigParams(1, false, matmul_tiling::ScheduleType::OUTER_PRODUCT,
            matmul_tiling::MatrixTraverse::FIRSTM, false);
        cubeTiling.SetMatmulConfigParams(matmulConfigParams);
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