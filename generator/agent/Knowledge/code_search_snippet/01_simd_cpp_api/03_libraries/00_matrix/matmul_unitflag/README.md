# Matmul 开启UnitFlag功能直调样例

## 概述
使能UnitFlag功能的Matmul样例，使样例中的CUBE计算流水与FIXPIPE数据搬出流水并行。使能UnitFlag功能后，在Matmul API内部实现MMAD指令和FIXPIPE指令的细粒度同步，从而使计算流水与数据搬出流水并行，提升样例性能

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── matmul_unitflag
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── matmul_unitflag.asc              // Ascend C样例实现 & 调用样例
```

## 样例描述
- 样例功能： 

  Matmul样例调用Matmul API计算时，通过配置MatmulConfig中的enUnitFlag参数为true，使能MDL模板开启UnitFlag功能。Norm模板、IBShare模板默认使能UnitFlag功能，MDL模板默认不使能UnitFlag功能。

- 约束条件
  - UnitFlag功能只支持Norm、IBshare、MDL三个模板。
  - 使能UnitFlag功能时，不支持样例内同时存在L0C搬出到Global Memory和L1搬出到Global Memory的两种流水。
  - 使能UnitFlag功能时，若同时使能L0C累加功能，不支持多次Iterate计算，一次GetTensorC输出。

- 样例规格： 

  本样例中：M = 1024, N = 4096, K = 1024。
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
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">matmul_unitflag_custom</td></tr>
  </table>

- 样例实现： 
  - Kernel关键步骤
    - 创建Matmul对象时，自定义MatmulConfig参数，将其中的enUnitFlag参数设置为true，使能UnitFlag功能，获得自定义的使用MDL模板的Matmul对象。
      ```cpp
      __aicore__ inline constexpr MatmulConfig GetUnitFlagCfg()
      {
          auto mmCfg = CFG_MDL;
      #ifdef ENABLE_UNITFLAG_FEATURE
          // enable UnitFlag feature
          mmCfg.enUnitFlag = true;
      #endif
          return mmCfg;
      }
      constexpr static MatmulConfig CFG_MDL_UNITFLAG = GetUnitFlagCfg();

      using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
      using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
      using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
      using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL_UNITFLAG> matmulObj;
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