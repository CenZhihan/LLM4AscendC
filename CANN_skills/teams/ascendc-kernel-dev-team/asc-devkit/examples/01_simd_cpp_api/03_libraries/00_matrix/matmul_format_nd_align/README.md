# Matmul ND_ALIGN格式输出场景直调样例
## 概述
在输入矩阵的N方向非对齐场景下，矩阵乘输出时使能N方向对齐的Matmul样例。通过设置矩阵乘结果C矩阵为ND_ALIGN格式输出，Matmul API将按照N方向32字节对齐的补齐规则输出C矩阵。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_format_nd_align
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_format_nd_align.asc              // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  Matmul样例调用Matmul API计算时，通过设置C矩阵的Format参数为CubeFormat::ND_ALIGN，使能矩阵乘输出的N方向32字节对齐功能，对输入的A、B矩阵做矩阵乘和加bias偏置。

- 样例规格： 

  本样例中：M = 128, N = 7679, K = 128。
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_format_nd_align</td></tr>
  </table>

- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象：C矩阵使能ND_ALIGN。
        ```cpp
        AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND_ALIGN, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>> matmulObj;
        ```

  - Tiling关键步骤
    - 创建一个Tiling对象：C矩阵使能ND_ALIGN。
      ```cpp
      cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
      matmul_tiling::DataType::DT_FLOAT);
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