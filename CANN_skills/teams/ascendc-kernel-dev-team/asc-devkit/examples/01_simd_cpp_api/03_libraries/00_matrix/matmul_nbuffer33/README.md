# Matmul 使用NBuffer33模板策略直调样例

## 概述
NBuffer33算法的Matmul样例，实现搬入搬出带宽平衡从而提升效率。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── matmul_nbuffer33
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_nbuffer33.asc    // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  Matmul样例使用模板参数MatmulPolicy配置为NBuffer33MatmulPolicy。单核计算的A矩阵切分为3x3个基本块，该3x3个A矩阵的基本块全载和保持在L1 Buffer中，每次与3x1个B矩阵的基本块计算矩阵乘，同时DoubleBuffer并行搬入下次计算所需的3x1个B矩阵基本块，直到singleCoreN方向的矩阵乘计算完成。

- 约束条件
  - 仅支持MDL模板。
  - 仅支持纯Cube模式（只有矩阵计算），暂不支持MIX模式（包含矩阵计算和矢量计算）。
  - 仅支持通过IterateAll接口获取Matmul的计算结果C矩阵。
  - stepM、stepKa、stepKb小于等于3。
  - 用户需要确保A矩阵全载的基本块大小与B矩阵载入的基本块大小之和不超过L1 Buffer大小。

- 样例规格： 

  本样例中：M = 256, N = 512, K = 192。
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_nbuffer33_custom</td></tr>
  </table>
- 样例实现： 
  - Kernel关键步骤
    - 具体步骤：
      - 创建Matmul对象，配置MDL模板和NBuffer33MatmulPolicy。
          ```cpp
          AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL,
                          AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                          AscendC::Impl::Detail::NBuffer33MatmulPolicy>
              matmulObj;
          ```

  - Tiling关键步骤
    - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等，开启NBuffer33模式。
      ```cpp
      matmul_tiling::MatmulConfigParams matmulConfigParams(1, false, matmul_tiling::ScheduleType::N_BUFFER_33,
                                                            matmul_tiling::MatrixTraverse::NOSET, false);
      tilingApi.SetMatmulConfigParams(matmulConfigParams);
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