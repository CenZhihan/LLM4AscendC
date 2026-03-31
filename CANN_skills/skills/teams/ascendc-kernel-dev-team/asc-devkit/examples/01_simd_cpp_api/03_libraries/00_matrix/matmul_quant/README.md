# Matmul算子反量化场景直调样例
## 概述
本样例介绍Matmul API实现int8类型输入、half类型输出的Matmul反量化场景的算子，支持同一系数的反量化模式和向量的反量化模式。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_quant
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_quant.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  Matmul算子调用Matmul API计算时int8_t类型输入，计算结果以half类型反量化输出，同时支持同一系数的反量化模式与向量的反量化模式。该场景下将C矩阵数据从CO1搬出到Global Memory时，会执行反量化操作，对输出矩阵的所有值采用同一系数或向量进行反量化。

- 算子规格： 

  本样例默认执行的算子shape为：M = 1024, N = 1024, K = 1024。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float16</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_quant_custom</td></tr>
  </table>

- 算子实现： 
  - Kernel实现
    - 计算逻辑：C = A * B + Bias。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[N, K]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。
    - 具体步骤：
      - 创建Matmul对象。
      - 设置左矩阵A、右矩阵B、偏置矩阵Bias。
      - 设置反量化参数。  
        当编译选项QUANT_MODE的值为1时，设置编译宏：CUSTOM_QUANT_VECTOR，编译执行向量的反量化模式。    
        根据是否定义宏：CUSTOM_QUANT_VECTOR，设置对应的反量化参数。
        ```
        #if defined(CUSTOM_QUANT_VECTOR)
            matmulObj.SetQuantVector(quantGlobal);
        #else
            float quantFloat = 0.1f;
            uint64_t quantValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&quantFloat));
            matmulObj.SetQuantScalar(quantValue);
        #endif
        ```
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
    - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
    - 获取Tiling参数的流程如下：
      - 创建一个Tiling对象。
      - 设置Matmul反量化模式。
        ``` 
        #if defined(CUSTOM_QUANT_VECTOR)
            tilingApi.SetDequantType(matmul_tiling::DequantType::TENSOR); // set TENSOR quant mode
        #else
            tilingApi.SetDequantType(matmul_tiling::DequantType::SCALAR); // set SCALAR quant mode
        #endif
        ```
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
  # -DQUANT_MODE=0：使能同一系数的反量化模式；
  # -DQUANT_MODE=1：使能向量反量化模式；
  # -m=0：使能同一系数的反量化模式；
  # -m=1：使能向量反量化模式；
  mkdir -p build && cd build;    # 创建并进入build目录
  cmake .. -DQUANT_MODE=0;make -j;    # 编译工程，以使能同一系数的反量化模式为例
  python3 ../scripts/gen_data.py -m=0   # 生成测试输入数据，以使能同一系数的反量化模式为例
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```