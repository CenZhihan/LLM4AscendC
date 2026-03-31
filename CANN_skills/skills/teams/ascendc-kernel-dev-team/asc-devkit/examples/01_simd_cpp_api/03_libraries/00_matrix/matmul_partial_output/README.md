# Matmul算子开启Partial Output场景直调样例
## 概述
本样例介绍Matmul高阶API实现开启Partial Output功能的单算子。Partial Output功能的应用场景为矩阵乘结果不需要累加，只需要输出baseM\*baseK和baseK\*baseN的计算结果baseM\*baseN。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_partial_output
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_partial_output.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  Matmul算子调用Matmul高阶API，开启Partial Output功能，对输入的A、B矩阵做矩阵乘。未累加结果搬运至UB Buffer（用户可在此阶段对数据进行自定义操作，例如反量化），最后对基本块进行累加得到最终结果。

- 算子规格： 

  本样例默认执行的算子shape为：M = 128, N = 128, K = 256。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_partial_output_custom</td></tr>
  </table>
- 算子实现： 
  - 约束条件
    - 仅支持MDL模板。
    - 仅支持调用Iterate和GetTensorC接口的连续写模式，不支持IterateAll接口以及非连续写模式。
    - 不支持带有Bias矩阵的Matmul计算，即不支持输入Bias矩阵。

  - Kernel实现
    - 计算逻辑：C = A * B。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
    - 具体步骤：
      - 创建Matmul对象。  
        创建Matmul对象时，自定义MatmulConfig参数，将其中的isPartialOutput参数设置为true，开启Partial Output功能，获得自定义的使用MDL模板的Matmul对象。
          ```
          __aicore__ inline constexpr MatmulConfig GetCustomMDLCFG()
          {
              auto mmCfg = CFG_MDL;
              mmCfg.isPartialOutput = true;
              return mmCfg;
          }
          constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG();
          AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CUSTOM_CFG_MDL>
              matmulObj;
          ```
      - 初始化操作。
      - 设置左矩阵A、右矩阵B。
      - 完成矩阵乘操作。  
        调用Iterate和GetTensorC接口的连续写模式。
          ```
          while (matmulObj.Iterate()) {
              matmulObj.GetTensorC(workspace[offset], 0, true);
              offset += tiling.baseM * tiling.baseN;
          }
          ```
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