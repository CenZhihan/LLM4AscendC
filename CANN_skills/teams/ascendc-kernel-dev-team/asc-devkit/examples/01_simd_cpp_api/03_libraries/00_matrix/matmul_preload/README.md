# M方向预加载Matmul样例
## 概述
M/N方向预加载的Matmul样例，可以减少MET2间隙。在MTE2流水间隙较大，且M/N数值较大时，可以开启对应的M/N方向预加载功能，提前加载A矩阵/B矩阵数据。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构
```
├── matmul_preload
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_preload.asc      // Ascend C样例实现 & 调用样例
```
## 样例描述

- 样例功能： 

  样例中实现了2种场景的样例，分别是Preload M方向流水并行，Preload N方向流水并行，通过MatmulConfig doMTE2Preload参数控制，即代码中的preloadMode取值为1时，M方向流水并行，代码中的preloadMode取值为2时，N方向流水并行。

- 约束条件
  - 预加载功能仅在MDL模板有效
  - 开启M/N预加载功能时需保证K全载，且M/N开启Double buffer
  - K全载的条件是singleK <= baseK * stepK
  - M开启Double buffer 的条件是depthA1 = stepM * stepK * 2
  - N开启Double buffer 的条件是depthB1 = stepN * stepK * 2

- 样例规格： 

  本样例中：M = 128, N = 24576, K = 512。
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_preload_custom</td></tr>
  </table>

- 样例实现
  - Kernel关键步骤
    - 配置MatmulConfig模板参数，开启doMTE2Preload开关，1（M方向）或 2（N方向），创建Matmul对象。
      ```cpp
      static constexpr MatmulConfig MM_CFG_PRELOAD = GetMDLConfig(false, false, 2); 
      AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>,
        MM_CFG_PRELOAD> matmulObj;
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