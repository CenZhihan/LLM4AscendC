# 分核计算bare_mix算子样例

## 概述

本样例介绍分核计算实现的CV融合算子bare_mix。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas 推理系列产品AI Core

## 目录结构介绍

```
├── bare_mix
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── bare_mix.asc            // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：

  本样例中实现的是[m, n, k]固定为[128, 128, 256]的bare_mix算子，仅在AIC核调用Matmul高阶API并计算，完成后在AIV核完成LeakyRelu的计算。

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">m*k</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">k*n</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">1*n</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">m*n</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">baremix_custom</td></tr>
  </table>

- 算子实现： 

  - Kernel实现  
    Matmul算子的数学表达式为：
    ```
    C = A * B + Bias
    ```
    其中A的形状为[128, 256]，B的形状为[256, 128]，Bias的形状为[1, 128]，C的形状为[128, 128]。

    LeakyRelu算子的数学表达式为：
    ```
    C = C > 0 ? C : C * S
    ```
    其中S为用户设置的LeakyRelu比例系数。
    
    本样例关键代码介绍如下：
    - 设置ASCENDC_CUBE_ONLY，仅在AIC核进行matmul计算
    - 设置Kernel类型为KERNEL_TYPE_MIX_XXX，同时启用AIV核和AIC核
    - 使用ASCEND_IS_AIC/ASCEND_IS_AIV隔离AIC/AIV核上的代码
    - 使用同步接口，自行完成核间同步
    ```c++
    #define ASCENDC_CUBE_ONLY //指定Matmul运行在AIC核上
    ...
    __mix__(1, 2) // 设置Kernel类型为KERNEL_TYPE_MIX_XXX
    ...
    if ASCEND_IS_AIC {
      ...
      // AIC核进行Matmul计算
      // AIC核完成计算后，通过AscendC::CrossCoreSetFlag<modeId, pipe>(flagId)发送同步flag
    }
    ...
    if ASCEND_IS_AIV {
      ...
      // AIV核通过AscendC::CrossCoreWaitFlag(flagId)接收同步flag
      // AIV核进行LeakyRelu计算
    }
    ```

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