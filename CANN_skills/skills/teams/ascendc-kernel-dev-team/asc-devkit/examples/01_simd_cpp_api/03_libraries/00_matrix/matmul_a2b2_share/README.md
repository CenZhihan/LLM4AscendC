# Matmul算子开启A2和B2全局管理直调样例
## 概述
本样例介绍调用Matmul API实现开启A2和B2全局管理的单算子。其中A2和B2的全局管理为，算子中所有Matmul对象共享A2和B2。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_a2b2_share
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_a2b2_share.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  Matmul算子算子中包含两次Matmul计算，两次Matmul计算有不同的左矩阵、右矩阵，相同的bias，通过设置每个Matmul对象中MatmulConfig的isA2B2Shared参数值为true，开启A2和B2的全局管理，即所有Matmul对象共享A2和B2，提高算子性能。

- 算子规格： 

  本样例默认执行的算子shape为：M = 7680, N = 480, K = 320。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a1</td><td align="center">M * K</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b1</td><td align="center">K * N</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">a2</td><td align="center">M * K</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b2</td><td align="center">K * N</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输出</td>
  <td align="center">c1</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  <td align="center">c2</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_a2b2_share_custom</td></tr>
  </table>

- 算子实现： 
  - Kernel实现
    - 计算逻辑是：Ascend C提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。Matmul的计算公式为：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 实现Matmul矩阵乘运算的具体步骤如下：
      - 创建MatmulConfig对象，配置NORM模板，设置isA2B2Shared参数值为true，创建两个Matmul对象。
        ```
        // In the first matmul calculation, `a1 * b1 + bias = c1`.
        AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
            matmulObj1;
        // In the second matmul calculation, `a2 * b2 + bias = c2`.
        AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, IS_TRANS_A>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, IS_TRANS_B>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL>
            matmulObj2;
        ```
      - 初始化操作。
      - 设置两次矩阵乘的左矩阵A1、A2、右矩阵B1、B2，共用Bias。
        ```
        matmulObj1.SetTensorA(a1Global);
        matmulObj1.SetTensorB(b1Global);
        matmulObj2.SetTensorA(a2Global);
        matmulObj2.SetTensorB(b2Global);
        if (tiling.isBias) {
            matmulObj1.SetBias(biasGlobal);
            matmulObj2.SetBias(biasGlobal);
        }
        matmulObj1.IterateAll(c1Global);
        matmulObj1.End();
        matmulObj2.IterateAll(c2Global);
        matmulObj2.End();
        ```
      - 完成两次矩阵乘操作。
      - 结束两次矩阵乘操作。

  - tiling实现
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
  python3 ../scripts/verify_result.py output/output1.bin output/golden1.bin output/output2.bin output/golden2.bin  # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```