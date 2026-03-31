# Matmul算子模板参数MatmulCallbackFunc直调样例
## 概述
本样例介绍Matmul API模板参数MatmulCallbackFunc的自定义使用方式。MatmulCallbackFunc用于配置左矩阵从Global Memory拷贝到A1、右矩阵从Global Memory拷贝到B1、计算结果从CO1拷贝到Global Memory的自定义函数，本样例以Global Memory自定义搬运到A1的回调函数为例，介绍该模板参数如何使用。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_callback
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_callback.asc              // Ascend C算子实现 & 调用样例
```
## 算子描述
- 算子功能： 

  Matmul算子对输入的A、B矩阵做矩阵乘和加bias偏置。将自定义的左矩阵搬运函数CustomDataCopyInA作为参数传递给Matmul的模板参数MatmulCallbackFunc，实现左矩阵从Global Memory到A1的自定义搬运，本样例以输入A矩阵为例，实现callback回调功能，对于输入B矩阵、输出C矩阵的callback回调功能也可以参考该样例的实现。

- 算子规格： 

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">(2560, 512)</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">(512, 128)</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">(128, )</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">(2560, 128)</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_callback_custom</td></tr>
  </table>

- 算子实现： 
  - Kernel实现
    - 计算逻辑：本样例的A矩阵为非连续搬运，每两个基本块需要跳转一次地址（即第一块与第二块连续排布，第二块与第三块间存在地址偏移，第三块与第四块连续排布，后续以此类推）。编写自定义回调函数前，需要确定切分后的SingleM、SingleK、baseM、baseK的大小及base块的分布。本样例中在Tiling侧设置单核计算量SingleShape：SingleM=128、SingleK=512、SingleN=128，然后在调测阶段调用GetBaseM、GetBaseK接口，打印出参数信息baseM=128、baseK=128，由此得知每个单核上有4个base块，用于Kernel侧的地址偏移计算，从而编写回调函数的搬运。变量offsetListGlobal保存单核上A矩阵的第0个base块和第2个base块的首地址，每个单核需要传入2个地址。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[1, N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 具体步骤：
      - 自定义左矩阵搬运函数CustomDataCopyInA，通过矩阵起始地址、偏移地址等，实现对左矩阵基本块baseM * baseK从Global Memory到逻辑位置A1的搬运。
      - 将自定义的CustomDataCopyInA传递给模板参数MatmulCallBackFunc，创建Matmul对象。
        ```
        AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>,
          CFG_NORM, AscendC::MatmulCallBackFunc<nullptr, CustomDataCopyInA, nullptr>> matmulObj;
        ```
      - 初始化操作。
      - 设置左矩阵A、右矩阵B、Bias。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

  - Tiling实现
      - Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
      - 获取Tiling参数的流程如下：
        - 创建一个Tiling对象。
        - 设置A、B、C、Bias的参数类型信息，M、N、Ka、Kb形状信息，SingleShape的M、N、K等。
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