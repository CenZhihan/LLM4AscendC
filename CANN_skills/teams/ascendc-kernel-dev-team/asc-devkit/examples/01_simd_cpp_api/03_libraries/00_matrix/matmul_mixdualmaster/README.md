# Matmul 双主模式直调样例
## 概述
使能双主模式（MixDualMaster）的Matmul样例，AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。

在Matmul API在默认的MIX模式下，Matmul API通过消息机制由AIV驱动AIC运行；而在双主模式下，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，性能更优。
当满足以下条件之一时，可以开启双主模式：
- 核函数类型为MIX，且AIC核数：AIV核数=1：1
- 核函数类型为MIX，且AIC核数：AIV核数=1：2，且A矩阵和B矩阵同时使能IBSHARE参数

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
## 目录结构介绍
```
├── matmul_mixdualmaster
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_mixdualmaster.asc              // Ascend C样例实现 & 调用样例
```
## 样例描述
- 样例功能： 

  本样例核函数类型为MIX，且AIC核数：AIV核数=1：2，且A矩阵和B矩阵同时使能IBSHARE参数，同一AIC对应的两个AIV的输入AB矩阵数据在L1 Buffer上相同。  
  通过设置MatmulConfig参数enableMixDualMaster使能双主模式。

- 样例规格： 

  本样例中：M = 12288, N = 256, K = 128。
  <table>
  <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float32</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">样例输出</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float32</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_mixdualmaster_custom</td></tr>
  </table>
- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象：调用GetNormalConfig接口将enableMixDualMaster参数设置为true，获取自定义模板MM_CFG，通过传入模板参数创建Matmul对象。
      ```cpp
      static constexpr auto MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, 
      true, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true, false); 
      // 倒数第二位参数enableMixDualMaster设置为true，创建Matmul对象。矩阵A矩阵B使能IBSHARE参数
      AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false, LayoutMode::NONE, true>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NONE, true>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, MM_CFG> matmulObj;
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