# Mx Matmul ScaleA转置直调样例

## 概述

本样例介绍了在Mx数据格式下，scaleA开启转置、scaleB不开启转置的带有量化系数的矩阵乘法，即MxMatmul计算场景。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_mx_scalea_trans
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_mx_scalea_trans.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能： 

  MatmulMxScaleaTransCustom算子调用Matmul API计算时，scaleA开启转置、scaleB不开启转置，左量化系数矩阵与左矩阵乘积，右量化系数矩阵与右矩阵乘积，对两个乘积的结果做矩阵乘法。

- 算子规格： 

  本样例中，算子实现支持的shape为：M = 32，N = 128，K = 128，scaleK为K整除32的结果4。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MatmulMxScaleaTransCustom</td></tr>
  </tr>
  <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M*K</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleA</td><td align="center">M*scaleK</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">true</td></tr>
  <tr><td align="center">b</td><td align="center">K*N</td><td align="center">fp8_e5m2_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">scaleB</td><td align="center">scaleK*N</td><td align="center">fp8_e8m0_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M*N</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_mx_scalea_trans_custom</td></tr>
  </table>
- 算子实现： 
  - Kernel实现
    - 计算逻辑：C = (scaleA ⊗ A) * (scaleB ⊗ B) + Bias。“⊗”表示广播乘法。
      - A、B为源操作数，A为左矩阵，形状为[M, K]，数据类型为fp8_e5m2_t；scaleA为左量化系数矩阵，形状为[Ceil(K/64), M, 2]，数据类型为fp8_e8m0_t；B为右矩阵，形状为[K, N]，数据类型为fp8_e5m2_t；scaleB为右量化系数矩阵，形状为[Ceil(K/64), N, 2]，数据类型为fp8_e8m0_t。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对(scaleA ⊗ A) * (scaleB ⊗ B)结果矩阵的每一行都采用该bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象：使用MatmulTypeWithScale使能scaleA、scaleB，并设置scaleA的SCALE_ISTRANS参数为true。
          ```
          typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false, AscendC::TPosition::GM, CubeFormat::ND, true> aType;
          typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, false, AscendC::TPosition::GM, CubeFormat::ND, false> bType;
          typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
          typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
          // 定义matmul对象时， 传入MatmulWithScalePolicy表明使能MxMatmul模板策略
          AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy> matmulObj;
          ```
      - 初始化操作。
      - 设置左矩阵A与左量化系数矩阵scaleA、右矩阵B与右量化系数矩阵scaleB、Bias。
          ```
          matmulObj.SetTensorA(aGlobal, isTransA);
          matmulObj.SetTensorB(bGlobal, isTransB);
          matmulObj.SetTensorScaleA(asGlobal, isTransScaleA);
          matmulObj.SetTensorScaleB(bsGlobal, isTransScaleB);

          if (tiling.isBias) {
              matmulObj.SetBias(biasGlobal);
          }
          ```
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
      - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。需要传入A/B/C/scaleA/scaleB矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
      - 获取Tiling参数的流程如下：
        - 创建一个Tiling对象：使用SetMadType使能Mx特性，使用SetScaleAType设置ScaleA的信息、使用SetScaleBType设置scaleB的信息。
          ```
          cubeTiling.SetMadType(matmul_tiling::MatrixMadType::MXMODE);
          cubeTiling.SetScaleAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, true);
          cubeTiling.SetScaleBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, false);
          ```
        - 设置A、B、C、Bias、scaleA、scaleB的参数类型信息；M、N、Ka、Kb形状信息等。
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