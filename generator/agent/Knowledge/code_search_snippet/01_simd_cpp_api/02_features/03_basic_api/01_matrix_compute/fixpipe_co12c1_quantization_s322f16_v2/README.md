# fixpipe_co12c1_quantization_s322f16_v2样例

## 概述

本样例介绍如何使用组合API Fixpipe或基础API DataCopy将矩阵乘的结果从L0C搬出到L1，并支持随路quant, relu能力组合，与unitFlag能力组合，输入int8_t类型数据，输出由int32_t类型量化为half类型。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── fixpipe_co12c1_quantization_s322f16_v2
│   ├── scripts
│   │   ├── gen_data.py                                   // 输入数据和真值数据生成脚本
│   │   └── verify_result.py                              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                                    // 编译工程文件
│   ├── data_utils.h                                      // 数据读入写出函数
│   └── fixpipe_co12c1_quantization_s322f16_v2.asc        // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  

  本样例中实现的是[M, N, K]固定为[128, 256, 128]的Matmul乘算子，对应的数学表达式为：
  ```
  C = A * B
  ```
  其中A的形状为[M, K], B的形状为[K, N], C的形状为[M, N]。

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A</td><td align="center">(128, 128)</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  <tr><td align="center">B</td><td align="center">(128, 256)</td><td align="center">int8_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">(128, 256)</td><td align="center">half</td><td align="center">NZ</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">fixpipe_co12c1_quantization_s322f16_v2</td></tr> 
  </table>
此用例支持配置使用AscendC提供的组合AIP Fixpipe或基础API DataCopy，用户可以通过配置USEDATACOPY=true使能基础API DataCopy实现矩阵搬出，与此同时此样例支持FIXPIPE指令多种随路能力组合：<br>
配置PREQUANTMODE=1设置为scalar量化模式，即整个C矩阵对应一个量化参数；<br>
配置PREQUANTMODE=2设置为tensor/vector量化模式，即C矩阵的每一列对应一个量化参数；<br>
配置PRERELUMODE=1设置C矩阵随路使能normrelu能力;<br>
配置ENUNITFLAG=true设置开启MMAD指令与FIXPIPE指令并行能力；<br>
注意MMAD与FIXPIPE需同时使能unitFlag, 当使能unitFlag时，L0C上的LocalTensor不能用[TQue](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0137.html)获取，需要改用[TBuf](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0161.html)。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  PREQUANTMODE=1 PRERELUMODE=0 ENUNITFLAG=false  USEDATACOPY=false 
  # PREQUANTMODE 1:SCALAR 2:VECTOR; PRERELUMODE 0:disable 1:Norm relu
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DPRE_QUANT_MODE=$PREQUANTMODE -DPRE_RELU_MODE=$PRERELUMODE -DEN_UNIT_FLAG=$ENUNITFLAG -DUSE_DATA_COPY=$USEDATACOPY;
  make -j;    # 编译工程
  python3 ../scripts/gen_data.py -pre_quant_mode=$PREQUANTMODE -pre_relu_mode=$PRERELUMODE; # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```